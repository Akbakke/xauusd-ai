# GX1 project state

Updated 2026-08-03.

## Decision

Status is **BLOCK**.

GX1 has a current offline V8/V13 dataset line, but the launch registry has no
admitted dataset, model or bundle. There is no completed unified smoke,
candidate, calibration, untouched-TEST direction edge, learned-sizing proof
or candidate-bound replay. Missing model evidence is an error, never `FLAT`,
`HOLD` or permission to load an older artifact.

## Current offline evidence

The executable handover validates these exact artifacts through the existing
train-launch contract before it reports them ready:

- V8 chronological dataset: 369,303 TRAIN, 5,904 VAL and 6,071 TEST rows;
- exact 513 signals, 142 continuous context fields and five categorical fields;
- schema-v3 V4 MTF cache with liveness PASS;
- M5/M15/H1/H4/D1, 111 fields per timeframe, eight families, 40
  family×timeframe routes and 555 feature×timeframe cells;
- V13 unified Exit lifecycle PASS;
- Entry M5 sequence 96 and Exit M1 sequence 480, with identical feature
  ownership, split boundaries and TRAIN normalization;
- immutable V15 smoke-recipe audit PASS.

The recipe is CPU-only: batch 8, six epochs, patience 3, 512 sampled TRAIN
rows, zero workers, two MTF layers, one specialist layer and explicit windows
`M5=16,M15=64,H1=96,H4=96,D1=252`. It is a trainability recipe, not an edge
claim. On 2026-08-03 the exact V15 launch arguments and all immutable bindings
passed the public `model-native-smoke-train --dry-run` route inside a 1 GiB
outer cgroup. No training or model output was started.

Exact paths and hashes are printed by `scripts/gx1_handover.sh`; they must not
be rediscovered by glob, mtime, version sorting or a `latest` pointer.

## Current model evidence

The 2026-08-03 six-epoch smoke was interrupted in epoch 3 by a Windows
bugcheck recorded as `HYPERVISOR_ERROR (0x20001)`. It published no completion
bundle or terminal checkpoint-admission event. All partial output is invalid
and cannot be resumed. The bugcheck identifies the host/Hyper-V failure class;
it does not prove that training caused it.

The previous one-epoch bounded attempt also failed closed: public direction
collapsed almost entirely to FLAT, auxiliary path skill remained near chance
and specialist/family cooperation gates were below contract. No best state or
bundle was admitted. These results diagnose trainability; they prove no
trading edge.

## Code-proven architecture

- One unique calibrated `LONG/SHORT/FLAT` argmax is the only Entry authority;
  an exact top tie fails closed.
- One unique calibrated `HOLD/EXIT_NOW` argmax from the same bundle/shared
  encoder is the only Exit authority; an exact top tie fails closed.
- The 513-field surface is 34 base plus 378 mandatory causal-layer outputs and
  101 deterministic TRAIN-only ranked fields.
- All eight specialists exist on every retained timeframe.
- Every 142+5 context field has one specialist owner.
- Direction uses one 26-group/96-value learned fusion.
- Path quality, MFE/MAE, tradability, survival, top/bottom timing, utility,
  Q/V/Advantage and learned size remain supervised model evidence, not second
  policies.
- Exit consumes the frozen Entry representation plus an exact contiguous,
  closed-M1 path. It is trained with Entry and cannot be attached or retrained
  later.
- Missing fields, schemas, hashes, outputs or class support fail closed.

This is source-contract proof only. It does not prove practical precision.

## Evidence retention

On 2026-08-03 the sole cleanup owner removed 33 exact superseded V21/V23/V26
leaves (5,447,068,479 bytes) after plan, dry-run, approval, same-device
quarantine and terminal revalidation. The four still-existing historical
registry references and every current V8/V13/V4/V15 anchor were outside the
target set. The legacy `entry_iql_runs` incident path was already absent by its
recorded 2026-07-07 deletion incident; this cleanup did not target it.

Terminal evidence is
`/home/andre2/GX1_DATA/reports/gx1_evidence_retention_cleanup_reports/OLD_RUNS_20260803_V1/GX1_EVIDENCE_CLEANUP_EXECUTION_20260803T153441676760Z.json`
with SHA-256
`f0e96fe751de8bcc25730d1a5bfa8939e2632a94d8523203d8ca52d932b9d99d`.
It has no dataset, model, direction or launch authority.

## Machine safety boundary

Every heavy producer, trainer, audit or replay must use
`scripts/gx1_capped_run.sh`. The hard limits are:

- one heavy GX1 job;
- at most 10 GiB job RAM and 512 MiB job swap;
- CPU affinity 0–1 and one numerical-library thread;
- at least 20 GiB host-available RAM before launch;
- verified cgroup limits and a host-wide nonblocking lock.

WSL is configured for 32 GB RAM and 4 GB swap, but the active VM still reports
about 43 GiB/8 GiB. Any model/dataset request above 4 GiB is therefore rejected
until a WSL restart applies the configured host cap. The handover reports
`wsl_vm_cap: PENDING_RESTART` and the 10 GiB runner exits 75. This guard must
not be bypassed or weakened.

## Remaining blockers, in order

1. Apply and verify the 32 GB/4 GB WSL cap after the next natural restart.
2. Complete one bounded V8/V13 smoke and publish a terminal immutable result.
3. Audit any produced smoke bundle for class/head/gate/Exit liveness and
   component movement.
4. Train one same-bundle candidate without touching TEST or replacing Exit.
5. Prove untouched-TEST direction, abstention, top/bottom alignment,
   path/utility, costs and supported slices.
6. Prove train==serve and influence for all required model routes.
7. Execute the exact candidate-bound full-TEST Entry/Exit replay.

No acceptance threshold is changed merely to make a run pass. The active
scope ends at offline train/OOS/replay; live, paper, broker, publisher,
promotion and drift work remain forbidden by `GX1_RULES.md`.

## Verification state

Focused handover, featurebase, lifecycle, trainer and safety tests are green.
The last complete repository-suite counts are historical source checkpoints,
not current model evidence. A new full-suite count is not claimed here.
