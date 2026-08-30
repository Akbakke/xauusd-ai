# GX1 binding rules

This file defines the only active project scope.

The current operational checkpoint is
[`docs/CURRENT_AUDIT_STATUS_20260828.md`](docs/CURRENT_AUDIT_STATUS_20260828.md).
It is intentionally stricter than historical run notes: background services
remain stopped and no full training, TEST, demo or live route is authorised
until the current V46 audit/preflight sequence is complete.

`bash scripts/gx1_handover.sh` owns the live candidate-session position. It
must verify the launch-state session reference, recipe/source-closure identity,
session contract and active checkpoint state at runtime; prose checkpoint
numbers are historical evidence only.
Only declared local runtime state and regenerable Python/pytest/ruff caches may
be ignored; any other ignored path blocks source identity.

The checkpoint-640 V4 candidate session is historical, source-bound evidence.
The new V5 current-source technical smoke recipe is a separate 32-row,
CPU-dry-run-only technical control; it has no CUDA, bundle or candidate authority.

> **2026-08-30 override:** the technical checkpoint bundle/reload parity and
> VAL-only label journal are plumbing evidence only.  They do not authorise
> external compute, a backtest/edge claim, TEST, OANDA, demo, paper or live
> activity. One freshly declared, source-bound guarded candidate
> learning-validation session has now exercised the repaired CUDA path.  Its
> first guarded window stopped normally at the 20-minute wall-clock boundary
> after 576 partial TRAIN batches.  A separate fresh-process canary then
> reloaded that exact state, resumed at batch 577 and durably checkpointed
> batch 640 before a controlled stop.  It did not reach first VAL (31,004
> full-TRAIN batches precede it).  Consequently it is operational and
> throughput evidence only — never a candidate, edge or backtest claim.

## Current performance and machine-safety boundary

- The only resumable candidate state is the source/recipe-bound session at
  partial TRAIN batch `640`; it has no VAL, TEST, bundle, PnL or backtest
  authority. A new source commit, recipe, output root or data identity must
  never be substituted for that state.
- Every heavy job enters through `scripts/gx1_capped_run.sh`: one job at a
  time, cgroup RAM `20G`, swap `512M`, CPU affinity `0-1`, numerical-library
  threads one and DataLoader workers zero. Adding CPU workers/cores is not an
  approved speed knob: measured data fetch is small compared with the GPU Exit
  calculation, while extra WSL CPU load increases the freeze/heat risk.
- CUDA is fail-closed at core `70 C`, actual draw `220 W`, resident VRAM
  `12 GiB`, one-second telemetry and a 20-minute process limit. The RTX 3090
  currently reports a configured driver limit of `390 W`; that is not a power
  cap or permission to draw 390 W. The actual-draw guard remains mandatory.
- CUDA activation retention is permitted only with the `0.45` allocator fence
  and deterministic FP32. TF32, autocast and compilation remain disabled. It
  reduced the measured 64-batch interval from `101.889 s` to `86.863 s`
  (14.7%). A full 31,004-batch TRAIN epoch is therefore about 11.7 hours of
  GPU work before VAL, excluding guarded preflight/restart overhead.
- Efficiency means resuming the exact hash-bound checkpoint, not rebuilding
  data or repeating broad audits. A full candidate should be planned either as
  controlled local resumptions or separately approved external compute; it may
  not weaken the guard, use more CPU opportunistically or touch TEST.

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
- Both native surfaces use the same ordered signal fields, continuous context
  and categorical context. **The composition is the owner tuples in
  `gx1/contracts/entry_model_native_signal_v1.py` and nothing here restates a
  count** (rule 4; on 2026-08-15 every count this file carried was stale).
  The shape is a frozen base block + the mandatory causal families + the
  complete code-owned candidate remainder. Entry reads `MODEL_NATIVE_SEQ_LEN`
  M5 bars; Exit reads 480 M1 bars
  and the latest 512 detailed path rows plus an all-time elapsed-bar feature
  and full-path hash chain; total trade duration is not capped. Exit also
  consumes the learned frozen Entry-decision token projected from the exact
  ordered six-block pre-argmax decision source; both widths are owned by
  `gx1/contracts/entry_decision_token_v1.py` and are not restated here.
- Each closed higher-timeframe lane has `MULTI_TF_FEATURE_COUNT_V4` ordered
  fields. Raw tick-volume
  primitives are computed by the one volume owner after OHLCV resampling with
  `volume=sum`; computed volume features are never resampled. The local volume
  window needs preceding rows, so each owner reads more native rows than it
  slices; the counts derive from the sequence length and the volume window.
  Missing warmup is an error, not a zero fill.
- The MTF matrix, cache manifest and full-input liveness bind the single UTC
  trading-session owner; their exact schema versions are printed by
  `bash scripts/gx1_handover.sh` and are never restated here. H4 bins open on 22/02/06/10/14/18 UTC and D1 opens at 22:00 UTC; the retired
  H4 00/04/... and calendar-midnight D1 grids are forbidden.
- Entry model inputs may come only from the exact hash-bound native M5 feature
  surface. It is loaded once and sliced by exact timestamp for TRAIN/VAL/TEST;
  split-local specialist recomputation, alternate M5 input lanes and soft
  alignment are forbidden. Exit analogously consumes the bound native M1
  surface plus its additive path.
- The eight specialists are structure, SMC/liquidity, trend, volatility,
  momentum, session/regime, chart geometry and price action/candles.
- Signal binds the exact causal candle geometry/relation/carry owner
  (`gx1/features/entry_candle_primitives_v1.CANDLE_PRIMITIVE_FEATURE_NAMES`,
  which owns its width) locally and per TF; `candle.raw_zero_range_flag` left
  it on 2026-08-15 and is exactly recoverable from the three surviving range
  shares. The retained six-field local SMC addition carries
  raw displacement, sided sweep depth, one-shot events and event age; these
  are evidence, not direction votes.
- Direction has one authority: unique argmax of the accepted model's
  `entry_action_q_bps` over the valid LONG/SHORT/FLAT actions — expected return
  in basis points, not a calibrated probability (`entry_fitted_q_v1.py`). Exit
  has one authority: unique argmax of the same bundle's `unified_exit_action`
  Q-values over HOLD/EXIT_NOW. A tie or missing evidence fails closed.
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
  data-learned. The objective and recipe owners
  (`gx1/contracts/entry_model_native_training_objective_v1.py`,
  `gx1/contracts/entry_model_native_train_recipe_v1.py`) are the only
  authorities; execute them for schema versions, keys and flags — nothing is
  restated here. Proven from source 2026-08-19: the sole decision loss is
  masked raw-bps MSE on fitted-Q, task weights are learned by trainable
  homoscedastic log-variance, and no cross-entropy holds decision authority
  (one masked BCE survives on the `trendline_event` auxiliary head). The
  retired description — "objective v6 / recipe v5", unweighted CE on the main,
  MTF and side classifiers, and a pending Wave-C audit of fixed magnitudes —
  described a system that no longer exists. Whether every static magnitude in
  the trainer is gone is **not examined**, so no such claim is allowed.
- The squeeze owner and exact six-clock manifest/materializer plumbing are
  production-integrated in source, and six per-clock TRAIN artifacts have been
  fitted and admitted since 2026-08-18. **The current set and its hash are not
  named here** (rule 13): several six-clock sets exist on disk with different
  `contract_sha256`, so resolve the binding from the run's own V4 cache
  manifest. M1/M5/M15/H1/H4/D1 each require their exact immutable TRAIN-only
  artifact before rebuild or use.
  Bare/default/cross-clock parameters are forbidden. Fit and serve must decode
  with one causal filter; two decoders in that owner is what made the
  2026-08-15 artifacts absorbing on all six clocks. No rebuild has yet been run
  on the new artifacts.
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
  both lanes over the closed window `[--registry-fit-train-start,
  --registry-fit-train-end]` with `--registry-fit-inner-end` strictly inside it
  (frozen with exact TRAIN-source provenance and a hash-bound pair-generation
  pointer into the build manifests; no default exists, and the chain proves the
  three boundaries equal its own split authority),
  builds both feature lanes, and passes both feature surfaces to
  preflight/rebuild. The retired event-local
  `canonical_features_v2.parquet`/legacy source-cascade route is forbidden.
- `--history-start` must precede `--train-start` by at least the widest per-TF
  receptive field in `PRODUCTION_MTF_PER_TF_WINDOW_BARS` — the D1 lane —
  counted as real closed D1 bars, never as calendar days. The chain enforces
  this at `model-source-identity` alongside the 96-row local M5 sequence
  warmup. The declared TRAIN window and its derivation are in
  `docs/TRAIN_WINDOW_WIDENING_20260819.md`; no window is restated here.
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
- Every capped trainer run must persist its child stdout/stderr in the
  pre-created immutable-adjacent sidecar log as well as its guard log. A bare
  `child_status` is insufficient failure evidence and does not authorize a
  retry.
- Delete generated runs only through the retention owner after reachability and
  active-process checks. Never delete unknown worktrees or user changes.
- Canonical CUDA's native guard stops above 70 C core, 220 W actual draw or
  12 GiB residency. WSL cannot set the host driver's physical lower power limit
  (the driver returns `Insufficient Permissions`), so 220 W is a one-second
  process stop, not a throttle. The two 2026-08-28 batch-32 V46 attempts reached
  71 C before a bundle. The repaired batch-8 32/32 smoke then completed four
  optimizer steps and validation within 65 C / 211.77 W / 8,751 MiB. Its active
  episode movement proof passed, but the bundle loader imposed a candidate-only
  Exit gate on smoke. Its next repeat passed that repair but exposed a stale
  Regime-FiLM metadata requirement. Later head-metadata repairs `e0cf52ed` and
  `64d648da` align and statically check trainer/loader keys. The final
  recipe-bound 32/32 smoke completed and atomically published its diagnostic
  bundle at 63 C / 212.37 W / 8,751 MiB. The exact guarded evaluator then
  completed frozen VAL predictions at 55 C / 156.03 W / 715 MiB. The repaired
  CPU smoke-bundle audit passes all data, feature, target, lineage and output
  checks, but blocks because three specialist gates never top-rank after four
  optimizer steps. Before any further CUDA step, the source tests and
  prediction-artifact/audit preflight must pass. Only a predeclared bounded
  learning-validation probe may follow; candidate admission remains strict and
  no other producer gets CUDA permission.
- An ephemeral remote GPU is permitted only for the same offline research
  scope, exact frozen commit and hash-bound V46 artifacts, after explicit
  operator cost approval. It must have automatic time/cost termination and no
  broker credential, demo, paper or live endpoint. Remote capacity does not
  waive smoke, bundle, VAL, TEST, economics or risk gates.

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
reference is the coin-flip null, not a dataset; its magnitude is substrate-
specific and belongs in `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`, never
restated here. V46 is the current explicitly hash-bound audited research
rebuild: it repaired the active M1 decision-to-fill causal targets but is not
admitted. The fitted-Q production-economics contract blocks activation, edge
claims and every paper/live route until immutable executable-price, cost,
financing, gap/terminal and portfolio evidence is bound. No recipe, dataset or
model is currently admitted.
