# Attended staged preflight safety contract

Status: implemented; test and fresh-recipe validation required before the
first observed V40 execution. The 2026-08-23 operator approval applies as a
standing safety constraint: no expanded model run may bypass these fixed
limits, active telemetry or the full data preflight. This changes neither the
dataset nor any execution authority.

> Current-status note, 2026-08-28: this document describes the historical
> attended-only exception. The active V46 canonical route is separate and is
> now limited to one fresh 220 W batch-8/32-row repeat after the prior smoke
> completed four steps but correctly rejected a retired static Exit path.
> Commit `a77ebb6e` repairs that proof. Do not use attended results, a 390 W
> configured driver limit or this document to bypass the canonical smoke/bundle
> gate. The authoritative current plan is the handover.

### 2026-08-26 operator-present 390 W amendment

The operator explicitly approved the attended-only actual-draw ceiling at
390 W. The current source-bound attended policy is therefore 390 W configured
and 390 W actual draw, while retaining the 70 C core stop, 12 GiB NVML-use
stop, one-second telemetry, 10 GiB cgroup, 512 MiB swap ceiling, two logical
CPU cores (affinity 0-1), and the staged 600+300-second deadlines. Canonical
training is separate: it accepts the observed 390 W configured limit but keeps
the stricter 70 C core, 220 W actual-draw and 12 GiB residency stops. The
Windows driver rejected a physical lower power limit from WSL; 220 W is thus a
one-second canonical process stop, not a throttle.

The fresh V46 run bound to source commit `42c78b94` completed the full data
preflight and two CUDA optimizer steps without a WSL/GPU reset, thermal
slowdown or a bundle/validation/TEST action. Its session contract and two-slot
checkpoint are technical trainability evidence only. Commit `d033a87f` adds a
durable guard sidecar for every future guarded attempt (start, telemetry,
stage transition, stop reason and exit status); it did not retroactively alter
the completed V46 session.

## The problem being solved

The current attended route retains a complete source-identity check even when
it uses the storage-only sequence-roll representation. That is correct: the
immutable sequence-roll audit must be re-hashed against the exact parquet and
manifest in the running process. The 2026-08-23 run reached
`MULTI_TF_DECISION_WINDOW_COVERAGE` and was then stopped by the 300-second
outer guard during that exact hash. It never allocated a CUDA model batch.

Changing the hash to a stat check, accepting an old audit without re-hashing,
sampling the normalizer, or letting a caller choose a larger timeout would all
make the result less trustworthy. None is acceptable.

## Proposed semantics

The attended-only route would receive two fixed, source-bound deadlines:

| Stage | Fixed maximum | Work included | Protection retained |
| --- | ---: | --- | --- |
| `data_preflight` | 600 seconds | exact parquet + manifest re-hash; full MTF/lifecycle validation; full TRAIN normalization; both datasets, target/contract and specialist-routing checks | 10 GiB hard cgroup, 512 MiB swap ceiling, two logical CPU cores (affinity 0-1), low priority, one-second GPU telemetry, 70 C core stop, 390 W actual-draw stop, 12 GiB NVML-use stop |
| `model_smoke` | 300 seconds | model construction, CUDA input-contract forward, optimizer and the declared attended smoke epoch | exactly the same controls, with a newly measured 300-second deadline |

The total maximum is therefore 900 seconds. It is a more tightly specified
attended diagnostic, not canonical training. The WSL `temperature.memory=N/A`
exception remains attended-only; it does not authorize canonical, candidate,
TEST, promotion, paper or live work. CPU temperature is still unavailable in
WSL and remains a residual operator-visible risk.

## One-way transition, not a timer bypass

The outer guard is the only timeout authority. Its initial state is
`data_preflight`; it monitors CUDA telemetry continuously even while GPU power
is low. It may transition exactly once to `model_smoke` only after an internal
notification from the canonical trainer.

The trainer notification must be emitted immediately before model construction,
after all of the following have succeeded in that same process:

1. V4 MTF cache load and full decision-window coverage.
2. Unified Exit lifecycle construction.
3. TRAIN and VAL sequence-roll audits re-hashed against their parquets and
   manifests.
4. Full TRAIN input-normalization fit over the exact lifecycle population.
5. VAL dataset/lifecycle binding, target/field contracts, dataloader shape
   proof, and exact eight-family specialist routing.

The notification must travel through a private, guard-created one-shot pipe
with an unpredictable token. The guard, not CLI arguments, creates its path
and token; the trainer gets them only through the protected environment. The
guard rejects an unexpected message, a duplicate message, an unreadable pipe
or any marker under canonical mode. The trainer's own unit test is responsible
for proving that its only emission site is after the exact pre-model boundary.
A caller-controlled file path, CLI flag, environment timeout or reusable
persistent readiness cache is forbidden.

On a valid transition the guard logs both elapsed preflight time and the new
model-stage deadline. It does not relax cgroup, CPU affinity, GPU thresholds,
telemetry frequency or process-group termination. A failure before the marker,
or no marker by 600 seconds, terminates the complete child process group as it
does today. A failure or timeout after the marker does the same at 300 seconds.

## Required proof before execution

Implementation is incomplete unless all of these pass:

1. Shell tests prove no-marker timeout, valid single transition, invalid-marker
   rejection, duplicate-marker rejection and model-stage timeout.
2. Unit tests prove the trainer can emit the marker only from
   `execution_tier=attended_only`, only after the exact pre-model contract
   boundary, and never in canonical execution.
3. The recipe audit source-binds every changed owner: capped runner, guard and
   trainer. Existing candidate/full routes reject every attended-stage input.
4. The whole suite passes under the attended 10 GiB / two-logical-core cap.
5. A fresh V40 recipe audit passes, followed only by an operator-approved,
   observed V40 attended run. No result from it may claim candidate, edge,
   OOS, TEST, PnL, paper or live authority.

After any completed trained candidate, the separate held-out gates still must
prove that every specialist encoder and each 5x8 family-by-timeframe route
changes decision margins. Staged preflight proves neither model influence nor
trading edge; it only lets the complete data path reach the model honestly.

## Resumable attended research session (implemented; bounded V40 session observed)

The observed V40 route can complete real optimizer steps inside the fixed
five-minute model phase, but a full smoke epoch contains more batches than one
such safety window. The trainer therefore owns one deliberately narrow
research-session mechanism:

- It applies only to `execution_tier=attended_only`, smoke profile, exactly one
  epoch and `grad_accum_steps=1`. Canonical and candidate paths neither create
  nor read it.
- Its fixed, source-owned budget is 60 **complete** optimizer steps. Its
  attended-only Exit forward is streamed in groups of 8 complete episodes;
  neither setting is exposed on the CLI. This is one operator-present session,
  still bounded by the guard-owned five-minute model deadline; it is not a
  candidate-training or edge-evaluation path. The outer guard remains active
  as a temperature, actual-power, telemetry and wall-clock backstop.
- Each completed step atomically writes a hash-bound state in the inactive one
  of two local slots, then atomically updates
  `ATTENDED_RESEARCH_SESSION_RESUME_POINTER.json`. This is resumable state,
  not evidence that a process is currently running; the guard sidecar's
  terminal record is the liveness authority.
  at it. State includes online/target model, optimizer, optional EMA/scheduler,
  exact batch permutation and CPU/CUDA/Python/NumPy RNG state. A process never
  resumes a partial gradient accumulation.
- The static contract binds the source commit, output name, full immutable
  TRAIN/VAL/M5/lifecycle bytes, normalization hash and all relevant smoke
  budget values. A source, data, recipe or output-name change rejects the
  existing session; it cannot silently start from a nearby artifact.
- State lives in a private sibling directory of the still-nonexistent bundle
  path. The bundle path remains absent. A completed session runs neither VAL
  nor checkpoint selection and publishes no bundle; its authority map fixes
  candidate, validation, TEST, promotion, paper and live to false.

This mechanism is not a relaxation of the resource policy. It does not raise
the 10 GiB cgroup, CPU affinity, 512 MiB swap ceiling, temperature limit,
actual-draw stop, configured power policy or five-minute model deadline. It
also does not introduce BF16, TF32, autocast or compilation: the current
training path remains deterministic FP32. Checkpoint state is deserialized to
CPU before it is restored to CUDA, avoiding a temporary second CUDA-resident
copy. Every real session execution must re-run the full source/data preflight
and the normal guard. A source/budget change deliberately rejects its prior
session rather than resuming it under changed memory behavior.

On 2026-08-24, the first session implementation (`44a253c6`) passed a fresh
immutable V40 recipe audit and persisted four complete, hash-verified
checkpoints (`complete=false`, `next_batch_offset=4`) in its private sibling
directory. No bundle was written, and no VAL, selection or authority-bearing
work ran. During that observed run the RTX 3090 remained below the guard's
core-temperature and actual-draw limits, but reached 24,260 MiB reported VRAM
usage and a new WSL/DXG `dxgkio_make_resident: Ioctl failed: -12` warning was
recorded. That is a platform-risk signal, not a success criterion. The former
two-step/32-row configuration was still unsafe under WSL. The current
60-step/8-row source-bound configuration follows a clean V46 two-step/8-row
execution that completed full data preflight and CUDA model stage without a
guard breach. It requires a fresh output path and recipe audit, must not resume
the older state, and is still not an established historical-CUDA
candidate-training path.

For attribution, the first batch now also emits `[TRAIN_PROFILE]` with Entry
online/target forwards, complete Exit training time, post-Exit backward time,
total synchronized wall time and peak CUDA allocation. The existing
`[UNIFIED_EXIT_PROFILE]` retains its materialize/online-Exit/target-Exit/
Bellman-backward breakdown. These are observability records only, not a speed
mode and not evidence of market edge.

## Operator decision and execution boundary

The operator approved implementation and the first observed bounded execution
on 2026-08-23. It extends an attended diagnostic from one five-minute
wall-clock to at most fifteen minutes, while CPU thermal telemetry is absent.
It does not authorize candidate training, edge claims, TEST, promotion, paper
trading or live execution. Those remain separate, evidence-gated decisions.

### Long-running historical CUDA training suspended

`--research-smoke` was removed after it kept nearly all 24 GiB VRAM resident
under WSL and the host/WSL session reset.  No historical CUDA train, bundle,
candidate, TEST, paper or live step may use that route. The no-data attended
hardware diagnostic remains available. A separate historical-data diagnostic
is admitted only through the attended-only route with its source-owned
low-VRAM geometry: CUDA batch size 8, one epoch, gradient accumulation 1, two
complete optimizer steps, at most eight 480-bar Exit episodes per backward
group, a 50% per-process CUDA allocator fence, and a 12 GiB NVML-use stop.
It also retains the five-minute model deadline, 70 C core stop, 390 W
**actual-draw** stop, 10 GiB cgroup, 512 MiB swap and CPU affinity 0-1. This is
one fresh bounded measurement, not permission for continued historical CUDA
training; it is non-promotable and may not create a bundle or any edge result.

The first measurement under this geometry reached the model boundary on
2026-08-24: full V40 data preflight passed, the allocator fence reported a
12,287 MiB budget, and a batch-8 Entry/Exit loss reached its first backward
pass. The outer guard then stopped the process for actual draw above 180 W
before an optimizer step completed. The intended bundle remained absent and
the session contains no completed checkpoint. This is a successful historical
safety stop, not trainability or edge evidence. It predates the explicit
2026-08-26 operator-present 390 W amendment above; no further power increase
or automatic retry is authorized by that amendment.
