# Candidate throughput decision — 2026-08-30

> **2026-09-03 re-entry note:** these are historical throughput measurements.
> V9 later completed one technical full TRAIN+VAL, but did not establish edge.
> The signed post-restart host bridge is currently unavailable (HTTP 503), so
> this document cannot authorise CUDA. See
> [`CURRENT_HANDOFF_20260903.md`](CURRENT_HANDOFF_20260903.md).

This decision is strictly pre-TEST.  It does not create activation, replay,
shadow, demo, paper, or live-trading authority.

## Scope and immutable inputs

All measurements below used the sealed pre-TEST V4 TRAIN/VAL artifacts, the
same source revision (`432f820b4dc4a3a2d4a12511e066009ba4e7bc1c` at the time
of measurement), deterministic CUDA execution, one 512-row TRAIN and
512-row VAL smoke pass, and no TEST artifact or metric access.  They were
executed serially under the canonical guarded envelope:

- CPU affinity `0-15`; cgroup memory `20 GiB`, swap `512 MiB`, tasks `128`;
- physical GPU power limit `210 W`; automatic stop limits of 70 C core,
  220 W actual draw, and 12 GiB VRAM;
- one heavy job only.

## Measured variants

| Variant | Immutable recipe SHA-256 | Train+VAL elapsed | Rows/s | Peak GPU | Decision |
| --- | --- | ---: | ---: | --- | --- |
| B8, workers 0 | `c5ec7eb85a7e79e4be4cf4576743500cff68b9dd5130f6fcffe2ff40ec530caf` | 137.183 s | 7.464 | 64 C, 187.79 W, 9,556 MiB | select |
| B9, workers 0 | `b3b968eb188b1c4ae6a2668db4d22e2f53b5351f09d1fa188814e2a398a5fec6` | 145.747 s | 7.026 | 64 C, 190.24 W, 9,742 MiB | reject: slower |
| B10, workers 0 | `0f86ba59e9fc63b5aef369a3ef6290c4a0335565188f8b9ee509bc07c9d96114` | 136.022 s | 7.528 | 64 C, 190.58 W, 9,720 MiB | reject: only 0.85% faster and changes the frozen B8 geometry |

The attempted B8/one-worker run was bound to immutable recipe
`d1ea68d917984e88d8c4f2bf689913c640a5824b806f9c43df1e91f589d164b4` and
ended before GPU training with
`[ENTRY_DATALOADER_WORKERS_INVALID] num_workers must equal 0 under the fixed low-memory recipe`.
This is an intentional trainer contract, not a resource failure.

## Selected candidate configuration

The next materialized candidate must retain batch size 8, `num_workers=0`,
full TRAIN population (248,028 rows), deterministic FP32 behavior, and the
frozen one-epoch candidate checkpoint policy. The one epoch is exactly 31,004
optimizer batches before its complete VAL pass; it must terminally complete
there rather than begin a second full TRAIN epoch. Batch 8 preserves that
explicit milestone, has the lowest measured VRAM use, and is within measurement
noise of B10's throughput without changing optimizer geometry.

The previously interrupted candidate remains preserved, resumable evidence at
batch 704 and is not reused as the selected current-source candidate.  Its
source binding predates the throughput-contract change.  The replacement must
be freshly materialized from a clean, committed source revision and receive a
new output bundle/session identity.

## Operational constraints

The GPU power cap must be verified as 210 W immediately before launch.  The
automatic cgroup and GPU guard remains authoritative; checkpoint/resume is for
recoverability, not permission to bypass a stop.  A completed B8 smoke is a
technical throughput measurement only: it makes no quality, edge, PnL,
promotion, or deployment claim.  TEST stays sealed until the separately
authorized post-candidate evaluation stage.
