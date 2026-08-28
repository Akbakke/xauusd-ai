# Current audit status — 2026-08-28

This document is the short, current-state override for operational decisions.
It complements the historical design documents; it does not grant execution
authority. `scripts/gx1_handover.sh` remains the executable status owner.

## Current verdict

The project is offline research only. There is no admitted model, predictive
edge, realised PnL, win rate, MAE/MFE result, calibration, untouched-TEST
result, demo account, paper route or live route.

Do not start full training, a rebuild, CUDA work, TEST evaluation, OANDA
activity, collector, Telegram notifier or dashboard from this state. The
collector/dashboard/notifier/self-check services are deliberately inactive.

The current V46 source data are retained and are not to be rebuilt merely to
obtain another smoke run. Historical liveness/normalization evidence predates
the repaired Exit MTF timing contract and is not candidate authority. It must
be regenerated from the retained immutable V46 inputs only after the remaining
static audit closes.

## What the current source now proves

- All eight families have exact local/context/MTF routing. Entry consumes local
  M5 and M15/H1/H4/D1 context; Exit consumes local M1 and all five MTF clocks.
  This is a structural/causal proof, not an edge result.
- A candidate export requires selected-checkpoint Entry VAL input-influence
  evidence across individual fields, all local/context families and all Entry
  MTF family routes. It also requires movement in every local and MTF family
  encoder component.
- Exit lifecycle loading recomputes the complete causally eligible Entry-row
  population from the immutable Entry and M1 clocks. A compact episode file
  cannot silently omit a serviceable episode.
- TRAIN normalization now binds the source-backed M5 feature surface and its
  exact sequence-reconstruction audit, in addition to the selected physical
  fit rows. A candidate cannot use an unaudited reconstructed sequence source.
- Recipe source closure now includes every executed Python package
  `__init__.py`. The trainer independently revalidates the byte-bound recipe
  at its own boundary and persists the identical recipe/source map in bundle
  metadata and lock; readiness and five-seed comparison reject omissions or
  split-brain provenance.
- The future serve adapter now opens only the exact V4 cache bound by the
  bundle, injects its frozen artifacts directly, and refuses a context snapshot
  from another immutable pair generation even if its timestamp is identical.
  Serve-parity evidence records the actual cache and full pair component hashes.
- The current offline boundary rejects direct paper-runner recovery and direct
  OANDA history ingestion before credentials, network, state reconciliation or
  writes. This preserves future code without granting present authority.
- The TEST evaluator is hard closed. A future one-time immutable TEST-release
  authority is required before any TEST file can be opened. This is intentional
  and does not block TRAIN/VAL audit work.

Recent repair commits: `3570ed51`, `64db63d1`, `a8717ec6`. The current
uncommitted audit repair set is the prerequisite for the next commit; no
dataset, CUDA or broker operation was started while applying it.

## Required next sequence

1. Finish the remaining read-only/source audit and update the explicit
   preflight checklist; do not add, remove or retune features.
2. Regenerate only the stale current-contract liveness/normalization evidence
   from V46 and run the bounded CPU preflight. Stop immediately on a mismatch.
3. Run one guarded, bounded learning-validation probe (one epoch over the
   agreed chronological research window) solely to prove end-to-end learning,
   field use and numerical stability. It is not a performance claim.
4. Inspect the resulting VAL research metrics and artefact proofs. Only if
   those are complete should a full candidate-training decision be considered.
5. Keep TEST sealed until a single candidate has passed VAL and an immutable
   release event is designed and reviewed. Demo/OANDA comes only after
   backtests and the separate executable-economics/risk gates.

## Safety boundary

Every audit/test runs through `scripts/gx1_capped_run.sh` under the audit cap.
No CUDA or producer job is authorised by this document. The host GPU power
limit is not a safety control from WSL; any future CUDA work must use the
existing attended telemetry guard and be manually observed.
