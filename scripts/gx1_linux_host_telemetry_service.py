#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


REQUEST_SCHEMA = "gx1_host_gpu_telemetry_request_v1"
RESPONSE_SCHEMA = "gx1_host_gpu_telemetry_v1"
TELEMETRY_PATH = "/gx1/v1/telemetry/"
ERROR_BODY = b'{"error":"telemetry_unavailable"}'
MAX_REQUEST_BYTES = 512
NONCE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_PATTERN = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
NVIDIA_QUERY = (
    "uuid,temperature.gpu,temperature.memory,power.draw,power.limit,memory.used"
)


class TelemetryUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceConfig:
    gpu_uuid: str
    private_key_path: Path
    nvidia_smi_path: Path
    openssl_path: Path = Path("/usr/bin/openssl")
    sensor_timeout_seconds: float = 2.0
    request_timeout_seconds: float = 2.0


def _require_regular_file(
    path: Path, *, executable: bool, private: bool = False
) -> None:
    if not path.is_absolute() or (private and path.is_symlink()):
        raise ValueError(f"unsafe path: {path}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ValueError(f"unavailable path: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"not a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"not executable: {path}")
    if private:
        if metadata.st_uid != os.geteuid():
            raise ValueError("private key must be owned by the service user")
        if metadata.st_mode & 0o077:
            raise ValueError(
                "private key permissions must exclude group and other access"
            )


def validate_config(config: ServiceConfig) -> None:
    if GPU_UUID_PATTERN.fullmatch(config.gpu_uuid) is None:
        raise ValueError("GPU UUID is malformed")
    if not 0.1 <= config.sensor_timeout_seconds <= 5.0:
        raise ValueError("sensor timeout must be between 0.1 and 5 seconds")
    if not 0.1 <= config.request_timeout_seconds <= 5.0:
        raise ValueError("request timeout must be between 0.1 and 5 seconds")
    _require_regular_file(config.private_key_path, executable=False, private=True)
    _require_regular_file(config.nvidia_smi_path, executable=True)
    _require_regular_file(config.openssl_path, executable=True)


def _finite_number(raw: str, *, positive: bool) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise TelemetryUnavailable("non-numeric sensor value") from exc
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise TelemetryUnavailable("sensor value is outside the protocol domain")
    return value


def read_gpu_telemetry(config: ServiceConfig) -> dict[str, object]:
    command = [
        str(config.nvidia_smi_path),
        f"--query-gpu={NVIDIA_QUERY}",
        "--format=csv,noheader,nounits",
        f"--id={config.gpu_uuid}",
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=config.sensor_timeout_seconds,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TelemetryUnavailable("nvidia-smi failed or timed out") from exc
    if result.returncode != 0:
        raise TelemetryUnavailable("nvidia-smi returned failure")

    rows = [
        row
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True)
        if row
    ]
    if len(rows) != 1 or len(rows[0]) != 6:
        raise TelemetryUnavailable("nvidia-smi did not return exactly one complete row")
    gpu_uuid, core_raw, memory_raw, draw_raw, limit_raw, memory_used_raw = (
        field.strip() for field in rows[0]
    )
    if gpu_uuid != config.gpu_uuid:
        raise TelemetryUnavailable("nvidia-smi returned the wrong GPU UUID")

    core_temp_c = _finite_number(core_raw, positive=False)
    memory_temp_c = _finite_number(memory_raw, positive=False)
    power_draw_w = _finite_number(draw_raw, positive=True)
    power_limit_w = _finite_number(limit_raw, positive=True)
    memory_used_value = _finite_number(memory_used_raw, positive=False)
    if not memory_used_value.is_integer():
        raise TelemetryUnavailable("memory usage is not an integer MiB value")

    return {
        "gpu_uuid": gpu_uuid,
        "core_temp_c": core_temp_c,
        "memory_temp_c": memory_temp_c,
        "power_draw_w": power_draw_w,
        "power_limit_w": power_limit_w,
        "memory_used_mib": int(memory_used_value),
        "observed_monotonic_ms": math.floor(time.monotonic() * 1000),
    }


def canonical_signature_payload(*, nonce: str, telemetry: dict[str, object]) -> bytes:
    return (
        f"{RESPONSE_SCHEMA}\n"
        f"{nonce}\n"
        f"{telemetry['gpu_uuid']}\n"
        f"{float(telemetry['core_temp_c']):.6f}\n"
        f"{float(telemetry['memory_temp_c']):.6f}\n"
        f"{float(telemetry['power_draw_w']):.6f}\n"
        f"{float(telemetry['power_limit_w']):.6f}\n"
        f"{int(telemetry['memory_used_mib'])}\n"
        f"{int(telemetry['observed_monotonic_ms'])}\n"
    ).encode("utf-8")


def sign_payload(config: ServiceConfig, payload: bytes) -> str:
    try:
        result = subprocess.run(
            [
                str(config.openssl_path),
                "dgst",
                "-sha256",
                "-sign",
                str(config.private_key_path),
            ],
            input=payload,
            capture_output=True,
            check=False,
            timeout=5.0,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TelemetryUnavailable("signature command failed or timed out") from exc
    if result.returncode != 0 or not result.stdout:
        raise TelemetryUnavailable("signature command returned failure")
    return base64.b64encode(result.stdout).decode("ascii")


def build_response(config: ServiceConfig, nonce: str) -> bytes:
    telemetry = read_gpu_telemetry(config)
    payload = canonical_signature_payload(nonce=nonce, telemetry=telemetry)
    response = {
        "schema_version": RESPONSE_SCHEMA,
        "request_nonce": nonce,
        "gpu_uuid": telemetry["gpu_uuid"],
        "core_temp_c": telemetry["core_temp_c"],
        "memory_temp_c": telemetry["memory_temp_c"],
        "power_draw_w": telemetry["power_draw_w"],
        "power_limit_w": telemetry["power_limit_w"],
        "memory_used_mib": telemetry["memory_used_mib"],
        "observed_monotonic_ms": telemetry["observed_monotonic_ms"],
        "signature": sign_payload(config, payload),
    }
    return json.dumps(response, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _strict_json_object(raw_body: bytes) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TelemetryUnavailable("duplicate JSON key")
            result[key] = value
        return result

    try:
        decoded = raw_body.decode("utf-8", errors="strict")
        request = json.loads(decoded, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TelemetryUnavailable("invalid request JSON") from exc
    if not isinstance(request, dict):
        raise TelemetryUnavailable("request JSON is not an object")
    return request


class TelemetryHTTPServer(HTTPServer):
    config: ServiceConfig
    request_timeout_seconds: float

    def get_request(self) -> tuple[socket.socket, Any]:
        request, client_address = super().get_request()
        request.settimeout(self.request_timeout_seconds)
        return request, client_address


class TelemetryRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "GX1HostTelemetry/1"
    sys_version = ""

    def _send_json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _unavailable(self) -> None:
        self._send_json(503, ERROR_BODY)

    def send_error(
        self,
        _code: int,
        _message: str | None = None,
        _explain: str | None = None,
    ) -> None:
        self._unavailable()

    def do_POST(self) -> None:
        try:
            if self.path != TELEMETRY_PATH:
                raise TelemetryUnavailable("invalid request path")
            if self.headers.get_all("Transfer-Encoding"):
                raise TelemetryUnavailable("transfer encoding is forbidden")
            content_types = self.headers.get_all("Content-Type", [])
            content_lengths = self.headers.get_all("Content-Length", [])
            if content_types != ["application/json"] or len(content_lengths) != 1:
                raise TelemetryUnavailable("invalid request envelope")
            try:
                content_length = int(content_lengths[0], 10)
            except ValueError as exc:
                raise TelemetryUnavailable("invalid content length") from exc
            if not 1 <= content_length <= MAX_REQUEST_BYTES:
                raise TelemetryUnavailable("request body is outside the size limit")
            raw_body = self.rfile.read(content_length)
            if len(raw_body) != content_length:
                raise TelemetryUnavailable("request body was truncated")
            request = _strict_json_object(raw_body)
            if set(request) != {"schema_version", "request_nonce"}:
                raise TelemetryUnavailable("request keys are not exact")
            if request["schema_version"] != REQUEST_SCHEMA:
                raise TelemetryUnavailable("request schema is invalid")
            nonce = request["request_nonce"]
            if not isinstance(nonce, str) or NONCE_PATTERN.fullmatch(nonce) is None:
                raise TelemetryUnavailable("request nonce is invalid")
            self._send_json(200, build_response(self.server.config, nonce))
        except Exception:
            self._unavailable()

    def do_GET(self) -> None:
        self._unavailable()

    def do_HEAD(self) -> None:
        self._unavailable()

    def do_PUT(self) -> None:
        self._unavailable()

    def do_DELETE(self) -> None:
        self._unavailable()

    def do_PATCH(self) -> None:
        self._unavailable()

    def do_OPTIONS(self) -> None:
        self._unavailable()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def create_server(
    config: ServiceConfig,
    *,
    listen_host: str = "127.0.0.1",
    port: int = 38128,
) -> TelemetryHTTPServer:
    validate_config(config)
    if listen_host != "127.0.0.1":
        raise ValueError("service must bind exact IPv4 loopback")
    if not 0 <= port <= 65535:
        raise ValueError("port is outside the valid range")
    server = TelemetryHTTPServer((listen_host, port), TelemetryRequestHandler)
    server.config = config
    server.request_timeout_seconds = config.request_timeout_seconds
    return server


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GX1 signed Linux GPU telemetry service"
    )
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=38128)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--nvidia-smi", default=Path("/usr/bin/nvidia-smi"), type=Path)
    parser.add_argument("--openssl", default=Path("/usr/bin/openssl"), type=Path)
    parser.add_argument("--sensor-timeout-seconds", default=2.0, type=float)
    parser.add_argument("--request-timeout-seconds", default=2.0, type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    os.umask(0o077)
    config = ServiceConfig(
        gpu_uuid=args.gpu_uuid,
        private_key_path=args.private_key,
        nvidia_smi_path=args.nvidia_smi,
        openssl_path=args.openssl,
        sensor_timeout_seconds=args.sensor_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    try:
        server = create_server(config, listen_host=args.listen_host, port=args.port)
    except (OSError, ValueError) as exc:
        print(f"FATAL: GX1 Linux host telemetry startup failed: {exc}", file=sys.stderr)
        return 78
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
