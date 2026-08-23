# Attended staged preflight: design, not authorization

Status: proposed only. This document does not change a runner, a timeout, a
dataset, a model recipe or execution authority. It exists because the current
single 300-second attended wall-clock has now stopped a real V40 run while it
was correctly re-hashing the 6.62 GB TRAIN parquet, before model construction.

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
| `data_preflight` | 600 seconds | exact parquet + manifest re-hash; full MTF/lifecycle validation; full TRAIN normalization; both datasets, target/contract and specialist-routing checks | 4 GiB hard cgroup, 512 MiB swap ceiling, one physical CPU core, low priority, one-second GPU telemetry, 75 C core stop, 250 W actual-draw stop |
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
with an unpredictable token. The runner, not CLI arguments, creates its path
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
4. The whole suite passes under the existing 4 GiB / one-core audit cap.
5. A fresh V40 recipe audit passes, followed only by an operator-approved,
   observed V40 attended run. No result from it may claim candidate, edge,
   OOS, TEST, PnL, paper or live authority.

After any completed trained candidate, the separate held-out gates still must
prove that every specialist encoder and each 5x8 family-by-timeframe route
changes decision margins. Staged preflight proves neither model influence nor
trading edge; it only lets the complete data path reach the model honestly.

## Explicit operator decision required

Do not implement or execute this design merely because this file exists. It
extends an attended diagnostic from one five-minute wall-clock to at most
fifteen minutes, while CPU thermal telemetry is absent. It requires a specific
operator approval for both implementation and the first observed execution.
