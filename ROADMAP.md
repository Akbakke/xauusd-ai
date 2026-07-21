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

Run lineage is XAU_SEQ513_REBUILD_20260720_V2. V1, V2, V3 and V4 event
attempts are non-authoritative failure evidence. Fresh V3 ranking, manifest
and preflight validated, but the V3 dataset builder stopped non-terminally
immediately after the canonical join while beginning Group-A causal context.
V4 then proved the capped runner is also not robust enough: the transient
service path failed without a user bus, and the restored scope runner lost the
fresh rank process before any ranking artifact/checkpoint existed. No dataset,
bundle, candidate, OOS edge or launch evidence is accepted.

## Ordered gates

1. Repair the rebuild execution path before another full run: every ranker and
   dataset-builder exit must write immutable terminal status/checkpoints, and
   Group-A must materialize inside its 30 GiB cgroup without changing feature
   semantics. Rebuild only from a fresh event root after that source repair.
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
