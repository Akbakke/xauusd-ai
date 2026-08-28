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
obtain another smoke run. Full-input liveness and its entire downstream
report-only chain have been refreshed from the retained immutable TRAIN/VAL
bytes. Historical normalization evidence still predates the repaired Exit MTF
timing contract and is not candidate authority; regenerate only that evidence
if a later, explicit candidate preflight requires it.

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
- The deep source audit traced the retained V46 feature chain end to end. Its
  latest liveness evidence covers 310 qualified local fields across all eight
  families on both native clocks (M5 Entry and M1 Exit), and the V4 cache
  covers all five MTF clocks with no constant field or exact duplicate pair.
  This proves wiring and variation, not predictive edge.
- The direct M1 fallback now derives the level/trendline registry clock from
  its declared local timeframe. Canonical V46 already supplied its exact M1
  layers, so this closes a latent fallback defect without rebuilding data.
- Report-only selective-edge summaries now read the emitted pre-registered
  metric scope and preserve boolean pass/fail values. This repairs reporting;
  it creates neither an edge claim nor TEST authority.
- CUDA-capable non-trainer paths now fail closed unless they are inside the
  capped/guarded producer scope; proof-only smoke-bundle audits and the
  synthetic Exit benchmark are CPU-only. Evidence cleanup checks for an open
  writer before moving any target into quarantine.
- A fresh V46 full-input-liveness v10 scan passed on 2026-08-28. It scanned
  all 248,028 TRAIN and 70,880 VAL rows, checked all 7,286,409,984 sequence
  values for finiteness and exact last-step/snapshot parity, and revalidated
  the complete five-clock MTF cache. It did not open TEST.
- The downstream V46 post-rebuild, smoke-manifest, smoke-readiness and
  trainability reports were then re-bound to that v10 scan. Their 6, 26, 6 and
  32 checks respectively passed with no failures. The final CPU-only recipe
  preflight rehashed the actual large TRAIN/VAL files and passed, while keeping
  TEST byte-opaque, CUDA unused and all trainer side effects false.

Recent safety/source commits: `34659e36`, `c3b67b6f`, `6ee59296`. The current
V46 report pointers in `PROJECT_STATE_xau_direction_launch.json` bind the new
v10 evidence chain. No dataset rebuild, CUDA, broker or TEST operation was
started while refreshing it.

## Required next sequence

1. Commit and validate the refreshed V46 report pointers. Stop immediately on
   any content, lineage or fail-closed-state mismatch; do not add, remove or
   retune features.
2. Inspect whether the preserved normalization evidence is stale for the
   current source commit. Regenerate only that report if required; do not
   rebuild V46 by default.
3. Only after that decision, obtain explicit authority for one guarded,
   bounded learning-validation probe (one epoch over the agreed chronological
   research window). It must prove learning, field use and numerical stability,
   not performance.
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
