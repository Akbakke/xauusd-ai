# GX1 XAUUSD model-native roadmap

Updated 2026-07-23. This is the execution roadmap, not launch authority.
PROJECT_STATE_xau_direction_launch.json remains BLOCK until every immutable
empirical gate below passes for one exact bundle.

## Objective

Build one XAUUSD model-native system that learns tops, bottoms and abstention
from the full stack and emits only calibrated LONG/SHORT/FLAT argmax. Trend,
session, structure, liquidity, volatility, momentum, price action, path
quality and utility remain learned inputs/targets; none may become a post-model
live rule or fallback.

## Current state

V24/V7 are the current immutable failed dataset/training lineage. V24 rebuilt
every source artifact from canonical XAU roots, including an event-local
Dec-2024 repair and an immutable live-M1 snapshot, through the last complete M5 bar at
`2026-07-22T12:05:00Z`. Its 393,176 x 188 FULL_PLUS source has 187 live numeric
fields, no constants, exact duplicates, non-finite values, stale self-paths or
fallback. The chain terminal is GREEN with SHA-256
`aaf5458fa53e83f16c436031650ff7ede322094b2376a9747fbe30f388891e48`.
That terminal is historical byte evidence, not current admission: the post-V7
audit proved signed dip-MFE target corruption and requires a new rebuild.

The V24 splits are 369,081 TRAIN / 5,904 June VAL / 4,115 July TEST. All
513+142+5 fields pass exhaustive liveness and all target/polarity checks pass.
Foundation feature, 46-target and specialist audits are PASS. All 479 selected
features route into the eight specialist contracts; TRAIN contains zero dead
signal fields, zero exact duplicate groups and zero unmapped signal/context
fields. One six-field exact duplicate group appears only in June VAL because
the short OOS window occupies one D1 regime state. It is retained as truthful
OOD evidence and cannot be used to weaken TRAIN policy.

The V22 audit had identified two exact TRAIN duplicate pairs between SMC
liquidity-pool proximity and S/R-memory proximity. The SMC features now blend
dedicated liquidity, recent swing and M5/M15/H1/H4/D1 level-cluster evidence,
while S/R keeps its separate repeated-level memory. V23 proved that separation
and the sparse-event floors, but smoke readiness caught an omitted
`iql_distillation=false` preflight key. V24 proves the complete six-key
side-effect map. The first V24 trainability review then caught a brittle
literal-source scan; commit `0f2b9468` now proves downstream consumers import
and use both exact signal-contract constants. The corrected immutable
trainability review is READY.

No accepted model has been produced on V24. The former smoke source blocker is closed:
commits `f08cd904`, `b5a61e21` and `bf5c61a0` provide one canonical
162-setting recipe owner/producer, validate the real pretrain schema, bind
executable source bytes and expose the exact post-smoke audit through the
single control surface. Six actual capped attempts then failed closed without
a bundle. V1 exposed a
static-versus-emitted aux-target mismatch, repaired by `9459babe`; V2 exposed
collapsed dataset-build/training-output IDs, repaired by `b986c8db`; V3
crossed both walls and completed five-timeframe prebuild before exposing the
trainer's false non-negative requirement for signed spread-aware MFE. Commit
`c9e2569f` also removes the related silent zero-clipping of signed MFE and path
quality in train/validation loss while retaining non-negative MAE. V4 then
built the full tensor surface and reached its first model forward, but the MTF
head incorrectly required a redundant `y_direction` batch alias instead of
canonical `y`; `f05b3390` fixes train and validation symmetrically without a
fallback. V5 then completed one full train/validation epoch with optimizer
steps. It preserved LONG/SHORT/FLAT prediction support, but direction slices
failed 23 checks and auxiliary tradable/bad-path AUCs were 0.509/0.482 versus
the fixed 0.52 floor, so checkpoint admission failed and no bundle was written.
V6 then completed six epochs. Epoch 4 briefly gave near-label global balance
and direction score 0.361111, but still failed 15 local slices and all required
auxiliary health. By epoch 6, LONG support was 0.058943, 29 slices failed, and
clean-edge/path-quality predictions had collapsed to Spearman +0.959 versus
only +0.699 between their VAL targets. No checkpoint or bundle was admitted.

Commit `37128985` makes exact epoch-wide specialist, timeframe and
family×timeframe gate health checkpoint-blocking at the unchanged 0.01 floor
and strengthens only the direction-neutral gate balance from 0.05 to 0.50.
Fresh readiness and trainability were READY under the pre-run contracts.
Recipe schema v2 binds
training run `XAU_SEQ513_SMOKE_20260723_V7` separately from dataset run
`XAU_SEQ513_REBUILD_20260722_V24`; recipe SHA-256
`fc012059594f5a197fdf145c86487e74ddfeba997f2604fa6759a0378416568d`
is PASS as immutable pre-execution evidence and its public dry-run passed.
V7 then completed six full TRAIN/VAL epochs before hard-red stopping emitted
`TRAIN_FAIL_NO_BEST_STATE`. Accuracy peaked at 0.403455 only through 85.1118%
FLAT; the final epoch predicted 71.4092% SHORT, failed 32 slices, retained weak
path auxiliary AUC, six cross-head collapses and starved specialist/family×TF
gates. No checkpoint or bundle was written.

The full audit in `PIPELINE_AUDIT_XAU_20260723.md` proves two P0s and multiple
P1s: signed dip-MFE clipping, selected-side bad-path LONG bias, replacement
sampling with only about 62% unique-row coverage, mismatched bidirectional aux
weights/metrics, partial checkpoint admission, incomplete MTF/scaler/context/
fusion identity and missing transactional launch authority at the audited
boundary. V21/V22/V23 large rejected split parquets have been removed while
their small terminal and audit evidence remains. No bundle, candidate,
untouched OOS edge or launch evidence exists.

The target/objective, sampler, conditional metrics, checkpoint influence,
complete physical-TRAIN normalization, all-147 context ownership, exact
five-timeframe cache bytes, positive TF scales, atomic bundle/event
publication, recursive active-Exit artifact identity, immutable approval,
identity-bound one-time vedtak, recoverable transactional launch finalization,
single-exposure enforcement, runtime lease recheck, broker/local XAU trade-ID
reconciliation and missing-trade-ID close path are now source-repaired with
regression proof. The finalizer uses the existing control surface and
canonical targets; it cannot create its own vedtak or accept alternate
authority roots. The re-audit found that the joint Exit finalizer does not run
the active models: it validates caller-supplied parquets. Launch/runtime now
reject that evidence outright, but a canonical full-TEST active-Exit producer
remains source P0. V24/V7 predate every one of these changes. No fresh artifact
or edge result exists.

The latest three-way audit also closes shared last-closed-M5 mapping, full
volume-prefix admission, XGB session/probability bridge validation, Exit V8's
173-field per-M1 cadence, V3 matrix/time/overlay/record storage semantics,
exact path-calibration supports and runtime-head prediction evidence V3.
Replay v6 separates canonical label outcomes from closed-bar decisions and
fresh active fills. Exact SourceTape open lookup, atomic frozen-pair loading
and an Exit-only pipeline factory now exist; the missing authority is the
full-TEST producer loop/event. No empirical artifact was created.

## Ordered gates

1. Preserve V24/V7 as immutable failure evidence. Do not rerun, patch data in
   place or promote any output.
2. Preserve completed source repairs: no-replacement sampling,
   bidirectional/conditional auxiliary evidence, exact recipe/M5/MTF bytes,
   complete TRAIN-fit normalization, 142+5 context ownership, all-head/group
   influence, atomic bundle/event publication, active-Exit byte identity,
   immutable approval/vedtak, transactional launch finalization, portfolio
   cap fail-close, broker/local trade reconciliation, runtime lease and
   execution fail-close, exact T+5/closed-M5 Exit timing, full V3 window
   coverage, transactional TradeState and production-only Exit loading.
3. Preserve the closed source contracts: canonical-v2 recomputes over complete
   verified native-M5 history; canonical-v3/BASE28 shares one atomic immutable
   generation identity; native-M5 has one closure/schema/hash owner; V3
   lineage binds the exact XGB bridge; and non-observable slippage-derived
   decision fields stay removed. ATR/ROC/VWAP, dependent normalized VWAP, SMC
   ATR and H1/H4 alignment use shared source owners and must not be forked.
4. Rebuild the per-bar Exit substrate at exact T+5, rescore V3 on exact row
   overlays and retrain Exit with checkpoint-bound features and one explicit
   serving fold. The retained research-only/non-production bundle is not an
   incumbent.
5. Extend the existing V3 dataset owner with the exact end-to-end
   training-dataset writer/event. It must derive all 173-field market rows,
   overlays and records from the bound sources; passing the strict
   reader/materializer is not producer authority.
6. Extend the existing sizing/replay owner with one canonical full-TEST
   producer around the existing exact `V12Pipeline.make_exit_decision`
   primitive and Exit-only frozen-pair factory. It must preserve the complete
   Entry snapshot, derive T+5 fill from hash-bound SourceTape, bind all
   canonical/BASE28/MTF inputs and active Exit bytes, and emit its own complete
   per-M1 actions/states/prices with zero fallback or horizon-cap pass.
7. Repair and prove canonical/live December-2024 M5 parity. The read-only
   audit found 3,430 impossible-geometry rows in both canonical M5 and
   live-prebuilt, including 2,799 weekend rows; the clean M1 supports 5,757
   rebuilt December buckets and leaves 3,459 canonical rows unbacked. The
   full loader also blocks on 2,375 invalid late-2024 prebuilt OHLC rows.
8. Rebuild fresh XAU-only splits, rerun every dataset/readiness audit and only
   then bind a new recipe. Compare full-history training with a declared
   TRAIN-only recent-regime challenger while preserving the final TEST window.
9. Train/evaluate a candidate only if smoke evidence is green; require OOS
   calibration, support, costs, TOP/BOTTOM timing, Q/V/Advantage, specialist,
   context and timeframe influence evidence.
10. Require candidate replay, exact serve parity, learned-sizing adoption with
   the active Exit stack, zero-order runtime parity, then the immutable
   adaptation/shadow lifecycle. Any missing or newer-red event remains BLOCK.

## Takeover

Run `bash scripts/gx1_handover.sh`, then read `AGENTS.md`,
`PIPELINE_AUDIT_XAU_20260723.md`, `SYSTEM_MAP.md`, this file and
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`. Never infer status from filenames,
partial artifacts or process absence.
