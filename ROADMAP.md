# GX1 XAUUSD model-native roadmap

Updated 2026-07-21. This is the execution roadmap, not launch authority.
PROJECT_STATE_xau_direction_launch.json remains BLOCK until every immutable
empirical gate below passes for one exact bundle.

## Objective

Build one XAUUSD model-native system that learns tops, bottoms and abstention
from the full stack and emits only calibrated LONG/SHORT/FLAT argmax. Trend,
session, structure, liquidity, volatility, momentum, price action, path
quality and utility remain learned inputs/targets; none may become a post-model
live rule or fallback.

## Current state

There is no accepted dataset or model lineage. V1-V13 are non-authoritative failure
or diagnostic evidence. V12 proved the full-history repair and built all three
splits, but was deliberately terminalized `ABORTED` before liveness PASS when
its 2026-06-14 cutoff was found stale. No dataset, bundle, candidate, OOS edge
or launch evidence is accepted.

V13 was rejected before dataset construction because its MTF cache used
`cv3_modelrange` rather than full canonical-v3. V14 rebuilt from a fresh event
and its source cascade is PASS through 2026-07-21T17:00Z: 392,959 rows x 188
columns, all 187 numeric fields live and full-v3 MTF identity proven. Its
ranking, seq513 dataset and all empirical gates are still pending. The chain
has no default end date.

The history-boundary repair is implemented but not yet exercised by a fresh
full run. Group-A now consumes an explicit full causal M5 prefix, validates
decision OHLC exactly and binds that prefix into checkpoint schema v2; live
HTF/regime state computes on the full prefix before slicing. The one chain
creates the immutable TRAIN-rank reference before
ranking and owns that reference through dataset audit; the capped runner
holds one host-wide exclusive heavy-job lock, Group-A persists exact
4096-row hash-bound chunks and can make one strict same-attempt resume, and
normal/signal exits publish immutable schema-v4 terminal events. The mandatory
specialist prefix is now 316 fields across eleven families, including exact M5
EMA50/200 state/cross evidence; only 163 fields are TRAIN-ranked. Ranking,
dataset and live use the same bound ECDF/ATR state and exact `close`/`atr`
inline source. V1-V11 remain failure evidence; their partial files are not
resume inputs.

## Ordered gates

1. Current-data V14 source/history cascade PASS. Run its fresh ranking/dataset
   chain and prove the exclusive runner, bounded checkpoints and immutable
   terminal event under the 30 GiB cgroup.
2. Accept only a fresh dataset whose split manifests, full-input liveness,
   target, specialist, leakage and pretrain audits all bind the same bytes.
3. Run smoke training, calibration and bundle audit with the exact recipe.
   Compare full-history training against a declared recent-regime adaptation
   phase; the trainer currently has no generic recency weighting, so no
   freshness claim is allowed until that candidate is implemented and wins on
   untouched OOS evidence.
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
