# Canonical host GPU telemetry bridge contract

Status: **retired as a canonical execution prerequisite on 2026-08-28**. The
bridge and Windows installer remain optional host-sensor tooling, not a gate
for offline research. Canonical CUDA now uses only the pinned, system-owned
WSL `nvidia-smi` path with a one-second guard: core <=70 C, actual draw <=250 W
and residency <=12 GiB. The observed 390 W *configured* driver limit is allowed
only because the actual-draw stop remains enforced. Neither path grants demo,
paper, live or production-edge authority.

## Historical bridge design (not a current CUDA prerequisite)

The canonical CUDA guard intentionally fails closed when any safety sensor is
unavailable. On the current RTX 3090 / WSL host, the system-owned
`/usr/lib/wsl/lib/nvidia-smi` reports a numeric core temperature, power and
VRAM residency, but `temperature.memory=N/A`. The Windows-interoperability
bridge is unavailable from this WSL instance. A real canonical probe on
2026-08-26 consequently exited 75 before Python, data loading or CUDA model
construction. The driver also reports a 390 W physical power limit, above the
canonical 250 W ceiling.

The attended-only diagnostic exception is deliberately not reusable here. It
is bounded, non-promotable research evidence only. A host bridge is a way to
make the existing canonical policy observable; it must never weaken that
policy.

## Sensor bootstrap (not a canonical shortcut)

The host bootstrap downloads the pinned official LibreHardwareMonitor release,
verifies its SHA-256 before extraction, and queries its `GPU Memory Junction`
sensor from an elevated native Windows process. It also records the native
`nvidia-smi` name, UUID and physical power limit. This is a zero-load
installation/probe step to determine whether the exact RTX 3090 has a numeric
VRAM-temperature source.

Its console JSON is validation evidence only. The output is deliberately
unsigned and nonces are not involved, so it cannot be read by the canonical
guard or be used as a file-based substitute for the future bridge. A successful
probe reduces the unknown hardware risk; it does not meet any of the bridge
admission criteria below.

## Staged bridge implementation (still inactive)

`scripts/gx1_host_telemetry_bridge_query.sh` generates a new 256-bit nonce for
each query, requires the exact request/response schemas, pins the expected GPU
UUID and public-certificate SHA-256, validates every numeric field, and uses
OpenSSL to verify the RSA-SHA256 response signature. Its deterministic tests
cover a valid response, wrong/replayed nonce, wrong UUID, excess field, bad
signature, bad certificate binding and timeout.

`scripts/windows/Install-GX1-HostTelemetryBridge.ps1` is the only proposed
host installer. It creates a non-exportable LocalMachine RSA certificate,
exports only its public certificate, locks its service tree to SYSTEM and
Administrators (Users receive read/execute only), and registers a SYSTEM task
which listens only at the fixed loopback endpoint. It returns live core/power/
VRAM-residency from Windows `nvidia-smi` and live `GPU Memory Junction` from
the already verified LibreHardwareMonitor library before signing all fields.
It never changes the GPU power limit.

The installer must be run and its reported public-certificate SHA-256 must be
committed as the source-bound client value. A signed Linux-to-host bridge probe
must then pass. Until those facts are observed, the guard's `unbound` sentinel
keeps the bridge path closed and these scripts are implementation material
rather than canonical safety evidence.

## Observed bootstrap evidence (2026-08-26; non-promotable)

The elevated host probe completed successfully against the installed,
SHA-verified LibreHardwareMonitor v0.9.6 release. Its interactive console
output reported the following values for the single pinned GPU:

| Field | Observed value |
| --- | --- |
| GPU name | `NVIDIA GeForce RTX 3090` |
| GPU UUID | `GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29` |
| VRAM junction temperature | `64 C` |
| Host power limit | `390 W` |

This proves that a host-only numeric VRAM-temperature source exists for the
right device. The output is an interactive installation probe, deliberately
not a signed bridge reply. It must never be admitted as canonical evidence.
The 390 W reading also means canonical CUDA remains blocked until the physical
limit is demonstrably at or below 250 W in a later signed bridge preflight.

## Historical bridge prerequisites (not current canonical constraints)

Before a canonical CUDA smoke can start, both of these must be demonstrated in
the same preflight:

1. The host driver's physical power limit for the pinned GPU is at most 250 W.
2. A trusted host observer returns a fresh numeric VRAM-temperature value for
   that exact GPU.

Core temperature, a power cap, low PyTorch allocator usage or an operator
watching a desktop dashboard are not substitutes for the second condition.

## Required request/response semantics

The future guard-owned bridge query is synchronous, not a file poll. For every
GPU guard sample, the Linux guard generates a fresh unpredictable 256-bit
nonce and sends one fixed request to a source-bound bridge endpoint. The host
returns exactly one signed response that includes:

| Field | Requirement |
| --- | --- |
| `schema_version` | Exactly `gx1_host_gpu_telemetry_v1`. |
| `request_nonce` | Exact byte-for-byte echo of that one guard request. |
| `gpu_uuid` | Exact match with the GPU UUID pinned in the canonical recipe/guard. |
| `core_temp_c` / `memory_temp_c` | Numeric finite Celsius values measured by the host. |
| `power_draw_w` / `power_limit_w` | Numeric finite watts measured by the host. |
| `memory_used_mib` | Non-negative integer measured by the host. |
| `observed_monotonic_ms` | Host-observed monotonic timestamp, fresh within the guard's fixed response-age allowance. |
| `signature` | Host private-key signature over every preceding response field, including the nonce. |

The guard embeds or source-binds the corresponding public key, uses a fixed
absolute endpoint and fixed timeout, and rejects malformed output, a bad
signature, a stale response, a duplicate nonce, a UUID mismatch, an extra
field, any `N/A`, an unavailable endpoint or a slow response. The request is
repeated at the existing guard interval and a single failed query terminates
the complete trainer process group.

The host signer must be a least-privilege, system-owned service. Its private
key and sensor source must not be writable by the WSL user, the trainer user,
the trading process or an interactive caller. The bridge service may use a
host-only sensor provider capable of reporting RTX 3090 VRAM temperature, but
the provider's raw shared-memory/file output is not itself a trusted transport.

## Explicitly forbidden shortcuts

- Accepting WSL's `temperature.memory=N/A` in canonical mode.
- A manually typed reading, environment variable, CLI value or caller-selected
  executable/path.
- A plain shared file, cached JSON/CSV, dashboard scrape or timestamp supplied
  by the caller.
- Continuing after a missed, stale, malformed or unsigned host response.
- Using attended-only evidence, thresholds or authority to satisfy canonical
  safety.

## Admission evidence for a future implementation

1. Deterministic tests cover valid response, bad signature, stale response,
   nonce replay, wrong UUID, every missing/non-numeric field, excess field,
   timeout and service disappearance. Each failure must kill the child group.
2. Recipe/source bindings include the bridge client, guard, public-key
   material, fixed endpoint contract and GPU UUID; ambient bridge controls are
   rejected by the trainer boundary.
3. A host-side installation test proves that an unprivileged WSL/trainer user
   cannot alter the signer, endpoint, key or reported sample.
4. An observed canonical preflight records numeric, signed VRAM temperature,
   physical power limit at or below 250 W and the exact sampled GPU UUID before
   it can load data or allocate a CUDA model.
5. Only then may the already prepared canonical batch-64 V46 smoke be
   executed. It remains a smoke; candidate, OOS, paper and live gates stay
   independent.

Until all five are met, the current fail-closed canonical guard is the correct
behavior.
