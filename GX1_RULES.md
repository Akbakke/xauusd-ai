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
- Both native surfaces use the same ordered 349 signal fields (30 base + 164
  mandatory causal/raw + all 155 code-owned candidates = 319 specialist fields over 11
  mandatory layer families; the counts derive from the owner tuples), 159
  continuous context fields and 5 categorical fields. This is signal v19 in
  direction contract mode v8. Entry reads 96 M5 bars; Exit reads 480 M1 bars
  and the latest 512 detailed path rows plus an all-time elapsed-bar feature
  and full-path hash chain; total trade duration is not capped. Exit also
  consumes the learned frozen 128-wide Entry-decision token projected from the
  exact ordered 609-wide six-block pre-argmax decision source.
- Each closed higher-timeframe lane has 171 ordered fields. Raw tick-volume
  primitives are computed by the one volume owner after OHLCV resampling with
  `volume=sum`; computed volume features are never resampled. The local volume
  window needs 95 preceding rows, so the Entry owner reads 191 native M5 rows
  before slicing 96 and the Exit owner reads 575 native M1 rows before slicing
  480. Missing warmup is an error, not a zero fill.
- MTF matrix V5, cache manifest v11 and full-input liveness v6 bind the single
  UTC trading-session owner. The signal split is v7 and mandatory stack v13.
  H4 bins open on 22/02/06/10/14/18 UTC and D1 opens at 22:00 UTC; the retired
  H4 00/04/... and calendar-midnight D1 grids are forbidden.
- Entry model inputs may come only from the exact hash-bound native M5 feature
  surface. It is loaded once and sliced by exact timestamp for TRAIN/VAL/TEST;
  split-local specialist recomputation, alternate M5 input lanes and soft
  alignment are forbidden. Exit analogously consumes the bound native M1
  surface plus its additive path.
- The eight specialists are structure, SMC/liquidity, trend, volatility,
  momentum, session/regime, chart geometry and price action/candles.
- Signal v18 uses the exact 26-field causal candle geometry/relation/carry
  owner locally and per TF. The retained six-field local SMC addition carries
  raw displacement, sided sweep depth, one-shot events and event age; these
  are evidence, not direction votes.
- Direction has one authority: unique argmax of the accepted model's calibrated
  LONG/SHORT/FLAT logits. Exit has one authority: unique argmax of the same
  bundle's HOLD/EXIT_NOW logits. A tie or missing evidence fails closed.
- No post-model handwritten direction/exit rule, threshold selector, fallback,
  cached decision, synthetic FLAT/HOLD, duplicate feature implementation or
  alternate replay route may affect the unique runtime argmax.
- The five handwritten regime composites, handcrafted `tf_agreement`
  objective/head and `signed_vol_z_20` are retired. Their genuine raw regime,
  trend, return and unsigned tick-volume evidence remains in the learned path.
- Position sizing is an auxiliary output trained only on its explicit tradable
  row mask against the frozen TRAIN-only selected-side path ECDF. It cannot
  influence or create direction and cannot create an order from FLAT/invalid.
- Runtime authority does not prove that every training-objective weight is
  data-learned. Objective v6/recipe v5 requires plain unweighted CE for the
  main, MTF and masked side classifiers and plain unweighted BCE for hierarchy
  binary tasks; Waves A/B retire direction and hierarchical distribution
  forcing. Fixed auxiliary task weights, rank margins and gate regularization
  remain a Wave-C audit, so no claim that all static magnitudes are gone is
  allowed.
- The squeeze owner and exact six-clock manifest/materializer plumbing are
  production-integrated in source. No production artifact has been fitted;
  M1/M5/M15/H1/H4/D1 each require their exact immutable TRAIN-only artifact
  before rebuild or use. Bare/default/cross-clock parameters are forbidden.
- Tick resolution is outside the current evidence surface. Exit remains closed
  native M1; no tick dataset, evaluation or trading claim is admitted.

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
  native M1/M5 from one pair manifest, TRAIN-fits the V29 registry state on
  both lanes from the explicit recipe input `--level-tol-quantile-q` (frozen
  with exact TRAIN-source provenance into the hash-bound build manifests; no
  default exists),
  builds both feature lanes, and passes both feature surfaces to
  preflight/rebuild. The retired event-local
  `canonical_features_v2.parquet`/legacy source-cascade route is forbidden.
- The level-registry runtime-population shadow replays the exact owner state
  machine only as a nonempty-support/provenance gate. It is neither another
  registry implementation nor authority for shadow or live trading.
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
- Ordinary audits/tests use at most 4G. The heavy offline dataset producers
  run as `--class producer` and may use at most 20G. The canonical trainer may
  use at most 20G (raised from 10G on 2026-08-09 on real batch=640
  measurement: pre-step host RSS baseline alone was ~10.1G, before any
  training step; see CLAUDE.md Host-capacity hard stop for the evidence).
  Never increase a cap as a workaround; misclassifying a heavy producer as an
  audit is a defect, not a reason to raise a ceiling — this was a correctly
  classified trainer job proven to need more headroom, not a misclassification.
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
Every artifact built on the retired 513 and 592 surfaces is invalid as
substrate for new training. The GREEN V28 (513) and sealed V29J (592) datasets
were retired on 2026-08-14 through the retention owner: no model, bundle,
calibration event or metric was ever derived from either, and the "frozen
comparison baseline" role they were given could never be executed, because
producing that arm requires training on a forbidden surface. The evaluation
reference is the coin-flip null (-13.16 bps TRAIN / -18.58 bps VAL), not a
dataset. The V30 rebuild is required; no recipe or model is currently admitted.
