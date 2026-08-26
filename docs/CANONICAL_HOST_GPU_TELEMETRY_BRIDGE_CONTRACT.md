# Canonical host GPU telemetry bridge contract

Status: the signed bridge remains a design-only prerequisite. The companion
`scripts/windows/Install-GX1-HostTelemetry.ps1` is admitted solely as an
elevated host sensor-installation and validation tool; it does not implement
the bridge, relax a guard, change a model/feature/target/dataset or grant any
execution authority.

## Why this exists

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

The host bootstrap installs the pinned `LibreHardwareMonitor.LibreHardwareMonitor`
Winget package and queries its `GPU Memory Junction` sensor from an elevated
native Windows process. It also records the native `nvidia-smi` name, UUID and
physical power limit. This is a zero-load installation/probe step to determine
whether the exact RTX 3090 has a numeric VRAM-temperature source.

Its console JSON is validation evidence only. The output is deliberately
unsigned and nonces are not involved, so it cannot be read by the canonical
guard or be used as a file-based substitute for the future bridge. A successful
probe reduces the unknown hardware risk; it does not meet any of the bridge
admission criteria below.

## Non-negotiable prerequisites

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
