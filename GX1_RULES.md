# GX1 binding rules

This file defines the only active project scope.

## One pipeline

```text
immutable OANDA XAU_USD M1 + M5
    -> same eight code-owned feature owners on separate native clocks
    -> Entry local M5 + M15/H1/H4/D1: LONG / SHORT / FLAT
    -> Exit local M1 + M5/M15/H1/H4/D1: HOLD / EXIT_NOW
    -> offline TRAIN / VAL / untouched TEST / same-bundle replay
```

- Entry and Exit use the same eight feature owners, formulas, ordered fields,
  TRAIN-only normalization and source lineage. Each owner computes native M5
  values for Entry and native M1 values for Exit; values are never copied
  between those clocks and there is no combined pre-owner M1/M5 package.
- Multi-timeframe candles must close before feature computation. Entry uses a
  local M5 sequence plus M15/H1/H4/D1 context. Exit uses a local M1 sequence
  plus M5/M15/H1/H4/D1 context. Resampling already computed M1 indicators into
  a higher timeframe is forbidden.
- One TRAIN-rank reference is fitted only from the immutable pair's canonical
  M5 `time/high/low/close/bid_close/ask_close` fields. The final Entry M5 model
  source must match those market values exactly from common-history start
  through TRAIN end. M1 and M5 consumers bind that same NPZ; fitting a second
  rank state or fitting from the downstream model source is forbidden.
- Both native surfaces use the same ordered 513 signal fields, 142 continuous
  context fields and 5 categorical fields. Entry reads 96 M5 bars; Exit reads
  480 M1 bars, capped at 512 path states, plus the frozen Entry representation.
- Entry model inputs may come only from the exact hash-bound native M5 feature
  surface. It is loaded once and sliced by exact timestamp for TRAIN/VAL/TEST;
  split-local specialist recomputation, alternate M5 input lanes and soft
  alignment are forbidden. Exit analogously consumes the bound native M1
  surface plus its additive path.
- The eight specialists are structure, SMC/liquidity, trend, volatility,
  momentum, session/regime, chart geometry and price action/candles.
- Direction has one authority: unique argmax of the accepted model's calibrated
  LONG/SHORT/FLAT logits. Exit has one authority: unique argmax of the same
  bundle's HOLD/EXIT_NOW logits. A tie or missing evidence fails closed.
- No handwritten direction/exit rule, threshold selector, fallback, cached
  decision, synthetic FLAT/HOLD, duplicate feature implementation or alternate
  replay route may affect a decision.

## Evidence rules

- Every consumed artifact is selected by explicit absolute path and SHA-256,
  never `latest`, mtime, glob order or a familiar run name.
- TRAIN alone may fit ranking and normalization. VAL may select/stop/calibrate
  only where its immutable contract says so. TEST remains untouched until the
  final candidate is frozen.
- M1/M5 source absence proven by the native OANDA authority is a market closure,
  not a bar to synthesize. Ordered observed rows advance through closures.
- Source, formula, schema, field order, signal-manifest hash, TRAIN-rank state,
  population, run identity and profile must match at every boundary. Any
  mismatch invalidates the full attempt.
- The only admitted dataset rebuild orchestration is the current-pair chain in
  `scripts/run_seq513_rebuild_chain_v1.sh`. It resolves canonical, BASE28 and
  native M1/M5 from one pair manifest, builds both feature lanes, and passes
  both feature surfaces to preflight/rebuild. The retired event-local
  `canonical_features_v2.parquet`/legacy source-cascade route is forbidden.
- No practical precision, win-rate or PnL claim exists without immutable,
  recomputable untouched-TEST and same-candidate Entry/Exit evidence.

## Frozen scope

Only offline source, featurebase, dataset, training, calibration, OOS and replay
work is allowed. Live, paper, demo, broker, daemon, publisher, live-tail,
promotion, drift adaptation and online weight updates are forbidden. Historical
modules cannot expand this scope and are not exposed by the control script.

Do not change architecture, add a feature family, create a compatibility lane,
remove samples/heads/features or alter objectives merely to make a run fit.
Complexity must live in the existing owners; unnecessary code is deleted.

## Capacity and cleanup

- Use `scripts/gx1_capped_run.sh` for every heavy producer, audit, train or
  replay. Run one job at a time on CPU cores 0-1 with 512 MiB swap.
- Ordinary audits/tests use at most 4G. The canonical trainer may use at most
  10G. Never increase a cap as a workaround.
- Feature producers run with exactly one worker. Model DataLoaders run with
  exactly zero subprocess workers. Canonical training is deterministic FP32;
  compile, autocast, TF32 and ambient fast-mode switches are forbidden.
- A cap kill, partial directory or interrupted event is failed evidence.
- Delete generated runs only through the retention owner after reachability and
  active-process checks. Never delete unknown worktrees or user changes.

## Takeover

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
```

Then read `AGENTS.md`, `SYSTEM_MAP.md`,
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` and the relevant code contracts.
The current V8/V13/V18 artifacts are stale under the repaired source/lifecycle
contracts. Rebuild is required; no recipe or model is currently admitted.
