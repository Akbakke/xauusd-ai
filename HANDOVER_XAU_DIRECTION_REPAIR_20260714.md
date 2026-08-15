# GX1 XAUUSD handover

Updated 2026-08-14. Run `bash scripts/gx1_handover.sh` before relying on this
document. `GX1_RULES.md` is binding.

## Current verdict

Launch remains `BLOCK`. The offline source architecture is connected, repaired
at the contract/source level and has focused capped-test evidence. No final
whole-repository result is asserted after the concurrent repair wave, and no
current-contract dataset, model, calibration, untouched-TEST edge, PnL or
win-rate proof exists.

State of the evidence chain:

- The current pair generation is
  `9b18e215061b0310bc0b9e962b00cfc2710f86e9484f3cee66f953f0077232cd`
  (published 2026-08-09, bootstrap mode; the 2026-08-04 parent generation
  `64d62c1f…a11b84c` is untouched history).
- Historical V28/V29J datasets were retired with their superseded feature
  contracts. They have no training, baseline or comparison authority.
- The current signal-v19/direction-mode-v8 V30 contract has 349 ordered signals:
  30 base, 164 mandatory causal/raw and all 155 code-owned candidates across 11
  mandatory families (319 specialist fields); context is 159 continuous plus 5
  categorical fields and every higher-timeframe lane has 171 fields. Handwritten
  scorebooks, five regime composites, the `tf_agreement` auxiliary
  objective/head and `signed_vol_z_20` were removed; genuine raw primitives,
  identified level/line state and causal events remain. No current-contract
  dataset has been built.
- Historical training-dynamics evidence: a V8-config smoke on the then-current
  substrate ended in total FLAT collapse; walk-forward probes refuted the
  snapshot direction edge and fixed the null-skill baseline (coin flip
  −13.16 bps TRAIN); seed variance flips collapse direction, so single-seed
  judging is invalid. The logit-adjusted CE and class-forcing recipe adopted at
  that time is now retired; those runs are history, not a current prescription.
  Objective v6/recipe-schema v5 uses plain unweighted CE for main, MTF and
  masked-side classification plus plain unweighted BCE for hierarchy binary
  tasks. Waves A/B retired direction and hierarchical distribution forcing.
  Fixed auxiliary task weights, rank margins and gate regularization remain
  for Wave C, and no materialized run recipe is admitted.
- Historical three-seed measurement 2026-08-12/13 (identical retired recipe,
  batch 64 x accum 10,
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
- signal v19/direction mode v8: 349 ordered signals (30 base + 164 mandatory
  causal/raw + all 155 code-owned candidates = 319 specialist fields over 11 mandatory
  families), 159
  continuous and 5 categorical context fields;
- per-timeframe V4 context width is 171, including raw volume and the five
  native-clock TRAIN-fitted squeeze-state fields plus
  trend/momentum event
  primitives, regime-flip flags and registry projections;
- signal v19 retains the exact 26-field causal candle
  geometry/relation/carry owner locally and per TF. The retained six-field
  local SMC addition emits raw displacement, sided sweep depth, one-shot
  events and event age rather than direction votes;
- MTF matrix V5, cache manifest v11 and full-input liveness v6 bind one UTC
  trading-session clock: H4 bins open on 22/02/06/10/14/18 UTC and D1 at
  22:00 UTC. Retired H4 00/04/... and calendar-midnight D1 caches are rejected;
  signal split v8 binds mandatory full-stack v13;
- V29 level and trendline registries (`gx1/features/level_registry_v1.py`,
  `gx1/features/trendline_registry_v1.py`) carry level identity, touch
  counts, ages, signed reaction history, break/retest events, sloped lines
  and channels; no precomputed confluence votes exist — fusion is learned;
- registry tolerances are TRAIN-fitted statistics: the rebuild chain requires
  the explicit recipe input `--level-tol-quantile-q` (adopted value 0.5,
  median, recipe owner `ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q`); the M5-lane
  constants freeze into the V4 cache manifest and the Exit M1-lane params
  freeze with exact fit-source provenance into the hash-bound M1-enriched
  manifest; both consumers fail closed without them. The post-fit level
  runtime-population shadow uses the exact owner state machine only as a
  nonempty-support/provenance gate; it is not a duplicate registry or a
  shadow/live route;
- the same volume owner computes `vol_z_20`, `vol_ratio_5_20` and `vol_pct_96`
  independently on each closed timeframe, after OHLCV resampling with tick
  volume summed. Computed features are never resampled;
- Entry: 191 native M5 source rows (95-row volume prefix) produce the exact
  96-row local slice plus leak-safe M15/H1/H4/D1 context;
- Exit: the same ordered fields on a 480-bar M1 local sequence plus leak-safe
  M5/M15/H1/H4/D1 context, frozen learned 128-value Entry-decision token and
  additive causal path; its source request is 575 native M1 rows, including
  the same 95-row volume prefix, with no zero-filled warmup. The token is the
  frozen 128-wide output of the exact learned 609-to-128 six-block pre-argmax
  Entry decision projection, not a generic pre-head embedding;
- closed OHLCV is built before each timeframe's features; finished M1 features
  are never resampled upward or copied into Entry;
- one shared encoder and one-bundle contract, with unique calibrated argmax for
  Entry (LONG/SHORT/FLAT) and Exit (HOLD/EXIT_NOW); ties and missing
  evidence fail closed.
- the five handwritten regime composites, `tf_agreement` auxiliary
  objective/head and `signed_vol_z_20` are retired. Raw per-TF regime class,
  EMA stack, trend age, D1 distance/rate of change, genuine change events,
  local return and unsigned tick-volume primitives remain for learned fusion;
- position-size targets are masked exact ECDF ranks of selected-side path
  evidence fitted only on TRAIN tradable rows. VAL/TEST apply the frozen ECDF;
  unmasked size training is forbidden and sizing has no direction authority.
- the TRAIN-fit squeeze owner and fail-closed six-clock artifact plumbing are
  production-integrated in source. No production artifacts have been fitted;
  a fresh M1/M5/M15/H1/H4/D1 fit and all downstream rebuilds are required;
- Exit remains native closed M1. No tick-resolution feature surface, dataset,
  OOS evaluation or trading claim exists.

## What is implemented

- native OANDA M1/M5 immutable source and pair contracts, with the current
  2026-08-09 pair generation published and hash-bound;
- the 2026-08-09 feature repair wave across all eight families (CLV
  recentering, USD→bps/ATR encodings, SMC backports, session fixes, dead
  column removal, routing fixes);
- the historical V29 Phase A addition, whose retained owners provide both
  registries, per-TF trend/momentum event primitives, regime-flip and
  swing-break events, and forward-realized aux rail labels replacing the old
  tautologies;
- V29 stage-3 prerequisites: the Exit M1-lane registry fit
  (`fit_v29_registry_m1_lane_params_from_m1`, same fit truths on the native
  M1 clock), chain plumbing of `--level-tol-quantile-q` and
  `--registry-fit-train-end` (defaults to the chain's `--train-end`; one
  origin), and lane-correct fail-closed resolution in both materializers;
- a contract requiring one immutable M5 Entry surface loaded once through
  bounded memmaps and shared as exact zero-copy timestamp windows across
  TRAIN/VAL/TEST;
- TRAIN-only ranking and normalization contracts; model-native Entry
  direction and unified Exit heads; M1 lifecycle builder/loader and
  same-bundle replay path; TRAIN-fitted/hash-bound Exit target horizon and
  executable-spread indifference policy (no CLI lookahead); immutable
  calibration provenance; learned sizing
  and serve/replay parity contracts;
- training-objective v6 metadata and 46-key recipe-schema v5: plain unweighted
  main/MTF/masked-side CE and plain unweighted hierarchy BCE, with direction
  and hierarchical distribution forcing retired in Waves A/B. Fixed auxiliary
  task weights, rank margins and gate regularization remain for Wave C;
- capped-run resource owner (4G audits, 20G producers/trainer, raised from
  10G on 2026-08-09 on real batch-640 RSS measurement) and immutable event
  machinery.

## What remains empirically unproven or unadmitted

1. The current-contract V30 dataset rebuild (registry fits, both feature lanes,
   lifecycle and splits) on the current pair generation.
2. Real-tape registry compute cost, event base rates, D1 warmup cost and
   liveness on declared TRAIN bytes (pre-adoption red gates).
3. Wave-C audit of fixed auxiliary task weights, rank margins and gate
   regularization, followed by an exact objective-v6/recipe-v5 run recipe.
4. Per-clock TRAIN squeeze artifacts and full adoption plumbing, if the owner
   passes its source/data gates.
5. A stable multi-seed smoke under the objective-v6 unweighted CE/BCE contract
   on the V30 substrate.
6. A full candidate trained on all TRAIN rows.
7. Immutable calibration using only its declared non-TEST split.
8. Untouched-TEST precision, PnL, drawdown and slice evidence, judged against
   the pre-registered protocol (walk-forward, coin-flip null −13.16 bps,
   ≥2 seeds and the abstention criterion).
9. Same-candidate unified Entry/Exit full-TEST replay and runtime parity.

Until all nine exist, practical precision and profitability are unknown.

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
2. Run the current-contract V30 rebuild chain
   (`scripts/run_seq513_rebuild_chain_v1.sh`) from the published pair with one
   new dataset run ID and the explicit
   `--level-tol-quantile-q 0.5`; the chain TRAIN-fits both registry lanes
   in-run and builds both native feature lanes one capped job at a time.
3. Admit the resulting M1/M5 surfaces, lifecycle and TRAIN/VAL/TEST only if
   all preflight/liveness/identity gates pass, including the registry
   pre-adoption gates (compute cost, event rates, warmup, liveness).
4. Close objective Wave C and squeeze adoption gates, then produce a distinct
   smoke training run ID and exact objective-v6/recipe-v5 capped run recipe.
5. Run multi-seed smoke (≥2 seeds; batch 64 × accum 10; unweighted
   CE/BCE contract) and audit every class, head, gate and Exit path.
6. If smoke passes, train one full candidate, calibrate and freeze it.
7. Open TEST once, evaluate under the pre-registered current-contract protocol
   and run the same bundle's unified Exit replay.

No architecture redesign is planned. Failures should be repaired in the
existing owner, with the smallest exact change that preserves the full model.
The squeeze owner is implemented but awaits adoption as stated above. Other
recorded backlog (session-anchored levels, VWAP events, box/candle rarity
gates and remaining fidelity work) stays outside the current rebuild until a
separate source/data-backed decision.

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
