# Canonical GPU telemetry — V9 signed Windows bridge contract

> Runtime-state rule: GPU telemetry does not identify a resumable candidate.
> `bash scripts/gx1_handover.sh` verifies the declared recipe/source closure,
> session contract, pointer and active state. TEST remains unread.

Status: **active V9 canonical CUDA prerequisite.** The V8 host hang proved that
WSL `temperature.memory=N/A` is insufficient for a 3090 candidate run. Every
canonical CUDA trainer, attended hardware smoke and allow-listed CUDA evidence
producer must instead obtain a fresh nonce-bound response from the signed
Windows host bridge. There is no native-WSL fallback and missing/non-numeric
memory-junction telemetry fails closed.

## Source-bound transport

`scripts/gx1_capped_run.sh` supplies, and
`scripts/gx1_guarded_trainer_exec.sh` verifies, all of the following values:

- query executable: `scripts/gx1_host_telemetry_bridge_query.sh`;
- WSL endpoint: `http://172.30.224.1:38128/gx1/v1/telemetry/`;
- public certificate:
  `/mnt/c/ProgramData/GX1/HostTelemetryBridgeV4/GX1HostTelemetryBridgePublic.pem`;
- public certificate SHA-256:
  `25c9260c2168db53cf58c5f963f2008d5163d80aa69699c5726e0680ed74eb6e`;
- expected physical GPU UUID:
  `GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29`.

The Windows bridge signs each response with a non-exportable local-machine RSA
key. The Linux query creates a new 256-bit nonce, verifies the certificate hash
and RSA signature, requires the exact response schema and GPU UUID, and returns
only numeric core temperature, memory junction, actual draw, configured limit
and VRAM residency. The Windows portproxy is bound to the WSL gateway only and
its firewall rule allows only the current WSL client address; it is not a LAN
listener.

## V9 immutable workload limits

For every CUDA tier the guard polls the signed host response before allocation
and once per second while the child process exists. It terminates the complete
process group when any check is unavailable or when one of these values is
exceeded:

- GPU core temperature: `65 C`;
- RTX 3090 GDDR6X memory junction: `80 C`;
- configured physical power limit: `160 W`;
- actual draw: `170 W`;
- resident GPU memory: `12 GiB`.

The Windows physical driver limit must be set to `160 W` and re-probed after a
Windows restart, driver reset or power interruption. The bridge installer never
changes that limit; `Install-GX1-HostTelemetry.ps1 -SetPowerLimitWatts 160`
does. A signed response at 2026-09-01 18:xx confirmed 47 C core, 52 C memory
junction, 39.76 W draw, 160 W limit and 403 MiB residency after the recovery.

V9 also binds candidate CPU affinity to `0-7` and all common numerical libraries
plus PyTorch to eight threads. This reserves eleven of WSL's nineteen logical
CPUs for the desktop/host. It is an affinity allocation, not an unavailable
cgroup CPU-rate controller. Memory remains hard-capped at 20 GiB, swap at
512 MiB, tasks at 128 and every candidate process has a two-hour wall-clock
limit with durable checkpoint/resume state.

## Admission consequences

V8's partial checkpoint is retained only as incident evidence and cannot be
resumed: its source closure used the unsafe 16-thread / 210 W / no-junction
policy. Before a new source-bound V9 candidate session, the exact V9 bounded
CUDA stability proof must pass. That proof grants no candidate, VAL, TEST,
paper, shadow, broker or live authority. A later full candidate still requires
a fresh immutable recipe, launch gate and all normal post-run audits.
