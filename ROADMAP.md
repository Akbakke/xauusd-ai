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

V24 is the current dataset/audit lineage. It rebuilt every source artifact from
canonical XAU roots, including an event-local Dec-2024 repair and an immutable
live-M1 snapshot, through the last complete M5 bar at
`2026-07-22T12:05:00Z`. Its 393,176 x 188 FULL_PLUS source has 187 live numeric
fields, no constants, exact duplicates, non-finite values, stale self-paths or
fallback. The chain terminal is GREEN with SHA-256
`aaf5458fa53e83f16c436031650ff7ede322094b2376a9747fbe30f388891e48`.

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

No model has been trained on V24. The former smoke source blocker is closed:
commits `f08cd904`, `b5a61e21` and `bf5c61a0` provide one canonical
162-setting recipe owner/producer, validate the real pretrain schema, bind
executable source bytes and expose the exact post-smoke audit through the
single control surface. Two actual capped attempts then failed closed before
training and created no bundle. V1 exposed a static-versus-emitted aux-target
contract mismatch, repaired by `9459babe`; V2 exposed that dataset-build and
training-output run IDs were incorrectly collapsed, repaired by `b986c8db`.
Fresh readiness and trainability are READY. Recipe schema v2 now binds
training run `XAU_SEQ513_SMOKE_20260723_V3` separately from dataset run
`XAU_SEQ513_REBUILD_20260722_V24`; recipe SHA-256
`0d28f1832f2602c7ae5ee5c461d645ad44c01382f8f9d4cac1354368626de37f`
is PASS and its exact public dry-run passes without creating a bundle. V21/V22/
V23 large rejected split parquets have been removed while their small terminal
and audit evidence remains. No bundle, candidate, untouched OOS edge or launch
evidence exists.

## Ordered gates

1. Preserve the PASS V3 recipe and exact V24 readiness/trainability bindings.
   Any copied dataset, stale schema, executable-source hash drift, dirty source,
   operator-supplied dataset identity or different source/smoke identity stops
   the path.
2. Run capped V3 smoke training and bundle audit with the exact produced
   recipe. Zero FLAT prediction support or passive head/specialist evidence is
   hard red.
   Compare full-history training against a declared recent-regime adaptation
   phase; the trainer currently has no generic recency weighting, so no
   freshness claim is allowed until that candidate is implemented and wins on
   validation and then untouched OOS evidence.
3. Train/evaluate a candidate only if smoke evidence is green; require OOS
   calibration, support, costs, TOP/BOTTOM timing, Q/V/Advantage, specialist,
   context and timeframe influence evidence.
4. Require candidate replay, exact serve parity, learned-sizing adoption with
   the active Exit stack, zero-order runtime parity, then the immutable
   adaptation/shadow lifecycle. Any missing or newer-red event remains BLOCK.

## Takeover

Run bash scripts/gx1_handover.sh, then read AGENTS.md, SYSTEM_MAP.md, this
file and HANDOVER_XAU_DIRECTION_REPAIR_20260714.md. Never infer status from
filenames, partial artifacts or process absence.
