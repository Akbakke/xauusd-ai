# GX1 XAUUSD handover

Updated 2026-08-11. Run `bash scripts/gx1_handover.sh` before relying on this
document. `GX1_RULES.md` is binding.

## Current verdict

Launch remains `BLOCK`. The offline source architecture is connected, repaired
and heavily tested, and one GREEN baseline dataset exists, but no model,
calibration, untouched-TEST edge, PnL or win-rate proof exists.

State of the evidence chain:

- The current pair generation is
  `9b18e215061b0310bc0b9e962b00cfc2710f86e9484f3cee66f953f0077232cd`
  (published 2026-08-09, bootstrap mode; the 2026-08-04 parent generation
  `64d62c1f…a11b84c` is untouched history).
- The V28 dataset chain (event root `XAU_ENTRY_EXIT_M15_20260809_V28`) ran
  GREEN end to end on the repaired feature substrate: 369,303 TRAIN / 5,904
  VAL / 6,551 TEST rows, TEST sealed. Per rule 7 those exact bytes are
  admitted evidence for the next gate only; they admit no model. V28 is the
  frozen comparison baseline for the pre-registered V29-vs-V28 evaluation.
- The 2026-08-11 event-gap review proved the pre-V29 surface carried almost
  no true level/break/retest events (~700 features, effectively one genuine
  break event). The V29 event surface is designed
  (`docs/V29_EVENT_SURFACE_DESIGN_20260811.md`), built and committed: level
  and trendline registries plus per-TF event primitives on all five
  timeframes. The signal contract is now 592 ordered signals = 34 base + 425
  mandatory causal + 133 TRAIN-ranked over 16 mandatory families (counts
  derive from the owner tuples). The V29 dataset rebuild has not run yet;
  every dataset built on the 513 surface is invalid as substrate for new
  training.
- Training-dynamics evidence to date: a V8-config smoke on the repaired
  substrate ended in total FLAT collapse; walk-forward probes refuted the
  snapshot direction edge and fixed the null-skill baseline (coin flip
  −13.16 bps TRAIN); seed variance flips collapse direction, so single-seed
  judging is invalid. The adopted repairs are logit-adjusted CE
  (`ENTRY_DIRECTION_LOGIT_ADJUST_TAU=1.0`, TRAIN priors, class weights 1.0)
  and mandatory multi-seed judging. Their validation resumes on the V29
  substrate.
- Three-seed measurement 2026-08-12/13 (identical recipe, batch 64 x accum 10,
  8 epochs, lr 3e-4, 25k rows): s1337 guard-OK 4/7 no collapse best 0.238;
  s1338 guard-OK 1/7 FLAT drift hard-red; s1339 guard-OK 4/6 best 0.256 then
  LONG collapse at epoch 6 hard-red — a limit cycle at a fixed step size.
  V30 package 5 adds two recipe-owned dampers:
  `ENTRY_TRAIN_LR_COSINE_DECAY=1` (the library cosine anneal over the declared
  epoch budget; a switch, no magnitude) and `ENTRY_TRAIN_WEIGHT_EMA_DECAY`
  (weight EMA read by validation/checkpoint selection). V30 package 6
  (2026-08-13) took the outstanding operator decision on the second one: the
  key declares a HORIZON, `epoch`, meaning one declared epoch of optimizer
  steps, so the recipe owner derives
  `decay = 1 - 1/ceil(train_rows/(batch_size*grad_accum_steps))` = **0.975**
  on the declared smoke budget (25k rows, batch 64, accum 10 -> 40 steps per
  epoch). `0.0` stays the exact-compatibility OFF sentinel. Env contract
  164 -> 166 pre-V29 keys. Neither damper has run end to end; existing
  readiness/recipe artifacts bind the old trainer bytes and correctly fail
  closed until re-materialization.
- The first four V29 chain attempts (2026-08-11) are immutable RED/ABORTED
  evidence, each repaired in the owning contract before relaunch: V29B —
  D1 `geomline_below_active` saturates at constant 1.0 on the declared
  population → presence-mask saturation contract in the V4 liveness owner
  (constant 0.0/dead siblings/silent events stay RED); V29C — the M5
  final-cache rebind published without the registry constants → bound
  through, plus a source-binding test; V29D — the flip-age fields' data-
  dependent warmup (NaN until each TF's first observed flip) reached the
  surface start → measured per-layer warmup floors trim the surface
  (declared in the manifest, bound by preflight and the dataset builder
  with full row accounting); V29E — the retention cleanup of superseded
  tape generations V1–V3 broke the successor ancestor chain → retired-
  ancestor retention attestation in the provenance owner (identity
  continuity via the executed DELETE_COMPLETE inventory; see
  docs/DATA_CONTRACT.md "Retired ancestors").

## Current feature architecture

- the same eight feature owners use one implementation each, run independently
  at native M5 for Entry and native M1 for Exit; no combined pre-owner M1/M5
  package;
- 592 ordered signals (34 base + 425 mandatory causal + 133 TRAIN-ranked,
  16 mandatory families), 142 continuous and 5 categorical context fields;
- per-timeframe V4 context width is 173, including trend/momentum event
  primitives, regime-flip flags and registry projections;
- V29 level and trendline registries (`gx1/features/level_registry_v1.py`,
  `gx1/features/trendline_registry_v1.py`) carry level identity, touch
  counts, ages, signed reaction history, break/retest events, sloped lines
  and channels; no precomputed confluence votes exist — fusion is learned;
- registry tolerances are TRAIN-fitted statistics: the rebuild chain requires
  the explicit recipe input `--level-tol-quantile-q` (adopted value 0.5,
  median, recipe owner `ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q`); the M5-lane
  constants freeze into the V4 cache manifest and the Exit M1-lane params
  freeze with provenance into the hash-bound M1-enriched manifest; both
  consumers fail closed without them;
- Entry: 96 local M5 bars plus leak-safe M15/H1/H4/D1 context;
- Exit: the same ordered fields on a 480-bar M1 local sequence plus leak-safe
  M5/M15/H1/H4/D1 context, frozen 128-value Entry representation and additive
  causal path;
- closed OHLCV is built before each timeframe's features; finished M1 features
  are never resampled upward or copied into Entry;
- one shared encoder, one committed bundle, unique calibrated argmax for
  Entry (LONG/SHORT/FLAT) and Exit (HOLD/EXIT_NOW); ties and missing
  evidence fail closed.

## What is implemented

- native OANDA M1/M5 immutable source and pair contracts, with the current
  2026-08-09 pair generation published and hash-bound;
- the 2026-08-09 feature repair wave across all eight families (CLV
  recentering, USD→bps/ATR encodings, SMC backports, session fixes, dead
  column removal, routing fixes);
- V29 Phase A event surface: both registries, per-TF trend/momentum event
  primitives, regime-flip and swing-break events, forward-realized aux rail
  labels replacing the old tautologies;
- V29 stage-3 prerequisites: the Exit M1-lane registry fit
  (`fit_v29_registry_m1_lane_params_from_m1`, same fit truths on the native
  M1 clock), chain plumbing of `--level-tol-quantile-q` and
  `--registry-fit-train-end` (defaults to the chain's `--train-end`; one
  origin), and lane-correct fail-closed resolution in both materializers;
- one required immutable M5 Entry surface loaded once through bounded memmaps
  and shared as exact zero-copy timestamp windows across TRAIN/VAL/TEST;
- TRAIN-only ranking and normalization contracts; model-native Entry
  direction and unified Exit heads; M1 lifecycle builder/loader and
  same-bundle replay path; immutable calibration provenance; learned sizing
  and serve/replay parity contracts;
- anti-collapse machinery with the grad-accum window buffer
  (`_PriorMatchAccumBuffer`), making the batch-640 statistical floor
  reachable at batch 64 × accum 10; logit-adjusted CE wired at every
  criterion site with recipe-owned tau;
- capped-run resource owner (4G audits, 20G producers/trainer, raised from
  10G on 2026-08-09 on real batch-640 RSS measurement) and immutable event
  machinery.

## What remains empirically unproven or unadmitted

1. The V29 dataset rebuild (registry fits, both feature lanes, lifecycle,
   splits) on the current pair generation.
2. Real-tape registry compute cost, event base rates, D1 warmup cost and
   liveness on declared TRAIN bytes (pre-adoption red gates).
3. A stable multi-seed smoke under logit-adjusted CE on the V29 substrate.
4. A full candidate trained on all TRAIN rows.
5. Immutable calibration using only its declared non-TEST split.
6. Untouched-TEST precision, PnL, drawdown and slice evidence, judged against
   the pre-registered protocol (walk-forward, coin-flip null −13.16 bps,
   ≥2 seeds, abstention criterion, V29 vs V28).
7. Same-candidate unified Entry/Exit full-TEST replay and runtime parity.

Until all seven exist, practical precision and profitability are unknown.

## Machine and process safety

Use `scripts/gx1_capped_run.sh`: 4G for tests/audits, at most 20G for the
heavy dataset producers and the canonical trainer (raised from 10G on
2026-08-09 after a real batch-640 RSS measurement; see CLAUDE.md), 512 MiB
swap, CPU 0-1 and one job at a time. Never increase limits to force progress.
A partial or killed run is failed evidence. The host has rebooted mid-run
twice on 2026-08-11; long-run logs belong under `/home/andre2/GX1_DATA/logs/`,
never `/tmp`.

The verified environment is CPython 3.10.12. `requirements.txt` contains only
the pinned direct runtime and verification packages needed for takeover.

A registered dirty nested worktree may belong to another agent; do not
inspect, clean or delete it. Preserve all unrelated user changes.

## Next implementation sequence

1. Verify the current commit with `scripts/gx1_handover.sh --check`.
2. Run the V29 rebuild chain (`scripts/run_seq513_rebuild_chain_v1.sh`) from
   the published pair with one new dataset run ID and the explicit
   `--level-tol-quantile-q 0.5`; the chain TRAIN-fits both registry lanes
   in-run and builds both native feature lanes one capped job at a time.
3. Admit the resulting M1/M5 surfaces, lifecycle and TRAIN/VAL/TEST only if
   all preflight/liveness/identity gates pass, including the registry
   pre-adoption gates (compute cost, event rates, warmup, liveness).
4. Produce a distinct smoke training run ID and exact capped recipe.
5. Run multi-seed smoke (≥2 seeds; batch 64 × accum 10; logit-adjusted CE)
   and audit every class, head, gate and Exit path.
6. If smoke passes, train one full candidate, calibrate and freeze it.
7. Open TEST once, evaluate under the pre-registered V29-vs-V28 protocol and
   run the same bundle's unified Exit replay.

No architecture redesign is planned. Failures should be repaired in the
existing owner, with the smallest exact change that preserves the full model.
Remaining recorded backlog (Phase B: session-anchored levels, VWAP events,
squeeze/box, candle rarity gates, vote removals; DST unification; D1
rollover) is deferred until the V29 evaluation verdict.

## Takeover

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
git worktree list --porcelain
```

Then read `GX1_RULES.md`, `AGENTS.md`, `SYSTEM_MAP.md` and
`docs/DATA_CONTRACT.md`. Use only `scripts/entry_next_edge_control.sh` for the
active workflow. It intentionally exposes no live or legacy replay route.
