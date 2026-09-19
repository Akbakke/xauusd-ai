# Canonical GPU telemetry — V9 signed Windows bridge contract

> **Authoritative status — 2026-09-08:** Read
> [`CURRENT_CLOUD_TRAINING_STATUS_20260908.md`](CURRENT_CLOUD_TRAINING_STATUS_20260908.md). The canonical
> handover currently returns `BLOCK`; the review hold is active, no trainer is
> running, and CUDA, TEST, paper/live and spending authority are all `NONE`.
> The cloud-control package is uncommitted and has known failing CPU tests.
> This block supersedes every lower runtime/status statement; lower dated text
> remains architecture, policy or historical evidence only.

> **Runtime-status rule:** the bridge contract remains required. A historical
> signed 160 W response is evidence only, not a standing host state: reboot,
> driver reset or a power-policy change invalidates it. CUDA remains fail-closed
> pending a clean preflight, a new signed response at 160 W, and explicit
> operator authority.
> Read
> [`CURRENT_HANDOFF_20260903.md`](CURRENT_HANDOFF_20260903.md) before action.

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

Nvidia treats the physical driver limit as runtime state, so it may reset after
a Windows restart, driver reset or power interruption. Run
`Install-GX1-HostTelemetry.ps1 -Install -SetPowerLimitWatts 160` once from an
elevated native Windows PowerShell. Besides applying the cap immediately, it
installs the `GX1GpuPowerLimit` SYSTEM startup task. That task waits for the
driver, reapplies the cap and verifies the exact GPU UUID plus a limit at or
below `160 W`; it then checks/reapplies every 15 minutes to cover a later
driver reset. It logs its result under `C:\\ProgramData\\GX1\\GpuPowerLimit`.
The bridge still independently observes and signs the physical state: automatic
reapplication does not make an old signed response valid after a restart. The
fresh source-bound query recorded on 2026-09-04 returned 52 C core, 56 C memory
junction, 39.05 W draw, 160 W limit and 392 MiB residency. It is host-safety
evidence only; the guarded launcher repeats the signed query immediately before
any CUDA allocation.

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
