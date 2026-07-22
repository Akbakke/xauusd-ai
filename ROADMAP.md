# GX1 XAUUSD model-native roadmap

Updated 2026-07-22. This is the execution roadmap, not launch authority.
PROJECT_STATE_xau_direction_launch.json remains BLOCK until every immutable
empirical gate below passes for one exact bundle.

## Objective

Build one XAUUSD model-native system that learns tops, bottoms and abstention
from the full stack and emits only calibrated LONG/SHORT/FLAT argmax. Trend,
session, structure, liquidity, volatility, momentum, price action, path
quality and utility remain learned inputs/targets; none may become a post-model
live rule or fallback.

## Current state

V19 is the first current-data lineage that reached a GREEN dataset-chain
terminal, but it is not accepted under the active contract. Source, TRAIN-only
ranking, exact 513 manifest, preflight, 369,081 TRAIN / 5,904 VAL / 3,934 TEST
rows, exhaustive 513+142+5 liveness and all 34 auxiliary-target checks bind
the same XAU tape and end at 2026-07-21T20:00:00Z. June/July H4/D1 volatility
shift is retained as `SHIFT_OBSERVED` evidence. The subsequent foundation
audit proved that all 57 required foundation fields were absent from V19's
selected tensor, so V19 is immutable superseded evidence only.

V20 then rebuilt all source from canonical roots through the last complete M5
bar at 2026-07-22T07:35:00Z. Its 393,122 x 188 FULL_PLUS source audit passed:
all 187 numeric fields were finite/live, with no constants, exact duplicates or
fallback. Fresh TRAIN ranking, the exact 513 manifest and preflight also
passed. Dataset construction failed closed before split publication because
`chart.geometry_channel_position_low_to_high` was used by structural
auxiliary-label construction but could still be removed by optional ranking.
V20 is terminal RED and no V20 artifact may be resumed or reused. There is no
accepted dataset, bundle, candidate, untouched OOS edge or launch evidence.

V18 is terminal RED because a host clock rollback made its ranking timestamp
temporally impossible. V1-V18 remain failure/diagnostic evidence and cannot be
resumed or promoted. The downstream audit found and repaired an impossible
smoke route: the new post-rebuild readiness producer consumes V19's exact
terminal/preflight/liveness/pretrain/split evidence and requires smoke to use
the same canonical dataset. A second audit removed an unreachable H1/H4
stateful branch whose shift semantics differed from the active array path.
Warmup is now explicit NaN and no completed HTF evidence fails closed.

The history-boundary repair was exercised successfully by V14 at full scale.
Group-A consumes an explicit full causal M5 prefix, validates
decision OHLC exactly and binds that prefix into checkpoint schema v2; live
HTF/regime state computes on the full prefix before slicing. The one chain
creates the immutable TRAIN-rank reference before
ranking and owns that reference through dataset audit; the capped runner
holds one host-wide exclusive heavy-job lock, Group-A persists exact
4096-row hash-bound chunks and can make one strict same-attempt resume only
before any immutable split/audit output exists, and
normal/signal exits publish immutable schema-v4 terminal events. The mandatory
specialist prefix is now 377 fields across twelve families: the 57-field
foundation cross-family layer, all prior specialist layers and exact M5
EMA50/200 state/cross evidence, plus every structural auxiliary-label signal
prerequisite. Only 102 fields are TRAIN-ranked. Ranking,
dataset and live use the same bound ECDF/ATR state and exact `close`/`atr`
inline source. V1-V11 remain failure evidence; their partial files are not
resume inputs.

## Ordered gates

1. Build V21 under `xau_seq513_model_native_direction_v3`, then run post-rebuild
   readiness plus foundation feature, target and specialist audits against its
   exact split bytes. V19 and V20 are immutable rejected/failure evidence.
2. Run smoke manifest, smoke readiness, trainability and recipe audits. Any
   copied dataset, stale schema, hash drift or different source/smoke identity
   stops the path.
3. Run smoke training, calibration and bundle audit with the exact recipe.
   Compare full-history training against a declared recent-regime adaptation
   phase; the trainer currently has no generic recency weighting, so no
   freshness claim is allowed until that candidate is implemented and wins on
   validation and then untouched OOS evidence.
4. Train/evaluate a candidate only if smoke evidence is green; require OOS
   calibration, support, costs, TOP/BOTTOM timing, Q/V/Advantage, specialist,
   context and timeframe influence evidence.
5. Require candidate replay, exact serve parity, learned-sizing adoption with
   the active Exit stack, zero-order runtime parity, then the immutable
   adaptation/shadow lifecycle. Any missing or newer-red event remains BLOCK.

## Takeover

Run bash scripts/gx1_handover.sh, then read AGENTS.md, SYSTEM_MAP.md, this
file and HANDOVER_XAU_DIRECTION_REPAIR_20260714.md. Never infer status from
filenames, partial artifacts or process absence.
