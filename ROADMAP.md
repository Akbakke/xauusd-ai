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

There is no reusable active run lineage. V1, V2, V3 and V4 event
attempts are non-authoritative failure evidence. Fresh V3 ranking, manifest
and preflight validated, but the V3 dataset builder stopped non-terminally
immediately after the canonical join while beginning Group-A causal context.
V4 then proved the capped runner is also not robust enough: the transient
service path failed without a user bus, and the restored scope runner lost the
fresh rank process before any ranking artifact/checkpoint existed. No dataset,
bundle, candidate, OOS edge or launch evidence is accepted.

Source repair is now implemented but not yet empirically exercised by a fresh
full run. The one chain creates the immutable TRAIN-rank reference before
ranking and owns that reference through dataset audit; the capped runner
holds one host-wide exclusive heavy-job lock, Group-A persists exact
4096-row hash-bound chunks and can make one strict same-attempt resume, and
normal/signal exits publish immutable schema-v4 terminal events. The mandatory
specialist prefix is now 316 fields across eleven families, including exact M5
EMA50/200 state/cross evidence; only 163 fields are TRAIN-ranked. Ranking,
dataset and live use the same bound ECDF/ATR state and exact `close`/`atr`
inline source. V3/V4 remain
failure evidence; their partial files are not resume inputs.

## Ordered gates

1. Source repair DONE; empirical execution proof pending. Rebuild from one
   fresh event root and prove the exclusive runner, bounded checkpoints,
   exact-resume path and immutable terminal event under the 30 GiB cgroup,
   without changing Group-A feature semantics.
2. Accept only a fresh dataset whose split manifests, full-input liveness,
   target, specialist, leakage and pretrain audits all bind the same bytes.
3. Run smoke training, calibration and bundle audit with the exact recipe.
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
