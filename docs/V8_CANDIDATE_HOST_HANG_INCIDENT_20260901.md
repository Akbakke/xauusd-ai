# V8 candidate host-hang incident — 2026-09-01

> **2026-09-03 re-entry note:** V8 remains incident evidence only. V9 later
> completed a technical TRAIN+VAL result, but after the latest restart the
> signed bridge query returned HTTP 503. New CUDA remains blocked; see
> [`CURRENT_HANDOFF_20260903.md`](CURRENT_HANDOFF_20260903.md).

Status: **V8 resume is blocked for host safety.** This is an operational
incident record, not candidate evidence and not an authority grant. TEST was
not read; there is no replay, shadow, paper, broker or live activity.

## What happened

V8 was the only full candidate launched from the immutable
`V8_ONE_EPOCH_CANDIDATE_20260901T080300Z_RECIPE.json` recipe. Its output bundle
was never published.

The trainer persisted checkpoint 51 at TRAIN batch 3,200 of 31,004:

- session pointer:
  `artifacts/.gx1-candidate-training-session.ENTRY_V8_ONE_EPOCH_CANDIDATE_20260901T080300Z_BUNDLE/CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json`;
- `next_batch_offset=3200`, `global_optimizer_steps=3200`, `complete=false`;
- checkpoint-state SHA-256:
  `75530f8574799f27d206f584189988eaf2c79ce4dbab6d0e5c85887ea705cfe8`.

The final checkpoint log record is at 13:19:17 local time. The final safety
heartbeat is at 13:20 local time. The operator lost contact with the Windows
host at 13:20 and had to remove power at 14:40. There is no guard `exit`
record, Python traceback, OOM event, final bundle or surviving trainer
process. The timestamp correlation establishes that this was a host hang while
V8 was running, not an expired Codex terminal or a normal bounded exit.

## Last observed workload state

The last guard heartbeat reported 64 C GPU core temperature, 188.05 W actual
GPU draw and 9,568 MiB resident VRAM. The trainer reported 4.18 GiB RSS. Those
facts rule out a triggered 70 C core, 220 W draw, 12 GiB VRAM or 20 GiB job-RAM
guard.

They do **not** prove that the workstation was safe:

- WSL reported `temperature.memory=N/A`; therefore the RTX 3090 GDDR6X memory
  junction was not observed by the canonical guard.
- The runner used CPU affinity `0-15` and 16 numerical threads on a host that
  exposes 19 logical CPUs (10 physical cores / 20 hardware threads). Affinity
  is not a CPU consumption limit.
- After the forced power cycle, the GPU's configured power limit had reset to
  390 W. The former 210 W driver limit was not persistent across the reboot.

The exact low-level cause cannot be proven from the pre-reset Windows logs:
the forced power removal reset WSL's kernel journal. A post-reset `dxgkrnl`
warning is evidence that the WSL GPU bridge is involved in the environment,
not proof of the original fault. The defensible diagnosis is a host-level
WSL/CUDA failure under this workload, with unobserved memory thermals and
unbounded CPU consumption as unresolved contributors.

## Verified containment gap

A no-load `systemd-run --user --scope -p AllowedCPUs=0-7 -p CPUQuota=800%`
probe still saw `cpuset.cpus.effective=0-18`, no effective `cpu.max`, and CPU
affinity `0-18`. The cgroup subtree enabled only `memory` and `pids`; it did
not provide a usable CPU controller to the unprivileged runner. Thus an outer
scope cannot narrow V8's hard-coded `0-15` affinity without changing the
source-bound runner or the WSL VM CPU exposure.

## Consequences and required gate

The V8 checkpoint remains immutable recovery evidence but is **not approved
for resume** with the same 16-thread / 210 W policy. Changing the runner's
CPU, power or memory-telemetry policy changes a file in V8's exact source
closure, which correctly makes that session non-resumable. A safe continuation
therefore requires a fresh source-bound candidate recipe and session after all
of the following are true:

1. a lower, persistent physical GPU power limit is independently verified
   after every Windows/driver restart;
2. the candidate policy has a real CPU bound rather than affinity alone;
3. missing 3090 memory-junction telemetry is either monitored by an
   independently verified host sensor or is a fail-closed candidate gate; and
4. a bounded CUDA stability proof passes under those exact new bounds before
   a full candidate is launched.

This preserves the V8 state for audit and prevents another silent host-risk
resume. It does not permit TEST access or any deployment action.
