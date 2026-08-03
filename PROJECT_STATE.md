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
- immutable V18 smoke-recipe audit PASS.

The recipe is CPU-only: batch 8, six epochs, patience 3, 512 sampled TRAIN
rows, zero workers, two MTF layers, one specialist layer and explicit windows
`M5=16,M15=64,H1=96,H4=96,D1=252`. It is a trainability recipe, not an edge
claim. On 2026-08-03 the exact V18 audit passed after binding memory-repair
commits `45421e70` and `98ea1c62`, the capacity runner and all
unchanged V8/V13/V4 artifacts. Its SHA-256 is
`818d8202bd0ab56a29fd43eea46e05bc2a9bfef285d811cd38a1e0909ca18285`.
No training or model output was started by the audit.

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

After the 32 GB/4 GB WSL cap became active, the first exact smoke launch
stopped before data loading because the capacity runner's four internal proof
variables collided with the trainer's ambient-control rejection. Commit
`667d704b` repaired that boundary. The following exact V16 launch validated
all V8/V13/V4 bytes, materialized TRAIN and VAL, bound all five timeframes and
the balanced unified Exit rows, and created the complete 7.55M-parameter
same-bundle model. At the first training step it reached the hard 10 GiB RAM
and 512 MiB swap ceilings and the cgroup killed only the job with exit 137.
The host remained healthy and no output bundle was created.

Commit `45421e70` keeps every feature, all four Exit samples per Entry row,
batch-8 objective semantics, dropout stream, head and gradient unchanged. It
recomputes each Transformer layer during backward instead of retaining every
Entry/MTF/Exit activation simultaneously; validation and inference keep the
ordinary path. The full 118-test model/trainer set passes under a 4 GiB cap,
including exact output and parameter-gradient equality. A full-size synthetic
batch with eight Entry rows, 32 Exit rows, 480 M1 feature bars and 512 path
bars completed forward and backward at 3,732,668 KiB peak RSS with zero swap.
The first V17 execution then exposed a separate preprocessing peak: one
8,192-row Arrow decode held float64 source values, a float32 cast and dirty
memmap pages together and was killed by the cgroup before training. Commit
`98ea1c62` fixes only I/O scheduling at 512-row decode batches with writeback
every 2,048 rows; row order and tensor bytes are unchanged. Its 91 focused
dataset/normalization/trainer tests pass under 4 GiB. V16/V17 are historical;
V18 is the sole current recipe authority. This proves bounded trainability of
one synthetic batch and bounded loader chunks, not a completed smoke or edge.

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
registry references and every then-current V8/V13/V4/V15 anchor were outside the
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

The active post-restart WSL VM reports about 31 GiB RAM and exactly 4 GiB swap,
matching the configured 32 GB/4 GB envelope within kernel accounting. A real
10 GiB/512 MiB no-op scope proved `memory.max`, `memory.high`,
`memory.swap.max` and `pids.max`. The guard must not be bypassed or weakened.

## Remaining blockers, in order

1. Complete one bounded V8/V13 smoke and publish a terminal immutable result.
2. Audit any produced smoke bundle for class/head/gate/Exit liveness and
   component movement.
3. Train one same-bundle candidate without touching TEST or replacing Exit.
4. Prove untouched-TEST direction, abstention, top/bottom alignment,
   path/utility, costs and supported slices.
5. Prove train==serve and influence for all required model routes.
6. Execute the exact candidate-bound full-TEST Entry/Exit replay.

No acceptance threshold is changed merely to make a run pass. The active
scope ends at offline train/OOS/replay; live, paper, broker, publisher,
promotion and drift work remain forbidden by `GX1_RULES.md`.

## Verification state

Focused handover, featurebase, lifecycle, trainer and safety tests are green.
The last complete repository-suite counts are historical source checkpoints,
not current model evidence. A new full-suite count is not claimed here.
