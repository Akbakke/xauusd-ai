# GX1 XAUUSD handover

Updated 2026-08-19. `scripts/gx1_handover.sh` is the executable status owner and
outranks this file — run it before relying on anything here. `GX1_RULES.md` is
binding scope; `CLAUDE.md` is the process constitution.

This file was 2,842 words on 2026-08-17; the chronological log of every chain
attempt, seed run and repair was cut and git holds it. Keep it under 1,800
words. A handover is a map of where you are, not a diary.

## Current verdict

Launch is `BLOCK`. There is **no admitted dataset, no model, no calibration, no
untouched-TEST result, no PnL and no win-rate proof** on the current contract.
V31 rebuild chains have run repeatedly since 2026-08-18 under
`/home/andre2/GX1_DATA/data/data/prebuilt/V31_CHAIN_*` and every one that
reached a terminal event ended RED; the newest has no terminal event at all, so
its partial output is invalid (rule 7). The surface has been materialised
several times and admitted zero times.

The evaluation reference is the coin-flip null and it is **substrate-specific**:
the −13.16 / −18.58 bps pair this file carried until 2026-08-19 was measured on
the retired V27 snapshot and does not transfer. On V31 bytes the reported
figures are null **−1.87 bps**, oracle **+23.84**, available skill **+25.71**
(2026-08-19) — *reported, not re-derived*: no hash-bound artifact under
`/home/andre2/GX1_DATA` carries them, so re-measure on the admitted rebuild
before any edge claim leans on them.

- **Source authority**: pair generation
  `9b18e215061b0310bc0b9e962b00cfc2710f86e9484f3cee66f953f0077232cd`
  (published 2026-08-09). Its 2026-08-04 parent `64d62c1f…a11b84c` is untouched
  history and is reachable from the current manifest — do not reclaim it.
- **Retired, no authority**: the V28 (513) and V29J (592) datasets. Nothing was
  ever trained on either, so neither can be the comparison baseline it was named
  as.
- **Seed variance flips collapse direction.** Single-seed judging is invalid: one
  three-seed run produced no-collapse, FLAT-drift and LONG-collapse from the same
  recipe.

## Current feature architecture

The same eight feature owners use one implementation each and run independently
at native M5 for Entry and native M1 for Exit. There is no combined pre-owner
M1/M5 package. Entry reads a local M5 sequence plus closed M15/H1/H4/D1 context;
Exit reads a 480-bar M1 sequence plus closed M5/M15/H1/H4/D1 context, the frozen
Entry-decision token and its causal in-trade path.

The surface shape is a frozen base block + the mandatory causal families + the
complete code-owned candidate remainder. **No width or schema version is restated
here** (rule 4/13) — the counts this file used to carry were stale by 88 fields
within two days, and the surface moved again on 2026-08-19 (schema `…_v34`
landed uncommitted, retiring duplicated and volatility-coupled fields; the newest
V31 signal manifest on disk already disagrees with HEAD's owner). Read them:

```bash
.venv/bin/python -c "import gx1.contracts.entry_model_native_signal_v1 as s, gx1.features.htf_features as h; \
  print('signal', s.MODEL_NATIVE_SIGNAL_DIM, s.MODEL_NATIVE_SIGNAL_SCHEMA_VERSION); \
  print('ctx', s.MODEL_NATIVE_CTX_CONT_DIM, s.MODEL_NATIVE_CTX_CAT_DIM); \
  print('mandatory', s.MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT); \
  print('candidates', s.MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT); \
  print('per_tf', h.MULTI_TF_FEATURE_COUNT_V4, h.HTF_V4_MATRIX_CONTRACT); \
  print('cache', h.HTF_V4_CACHE_SCHEMA_VERSION, h.HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION); \
  [print(' fam', n, len(f)) for n, f in s.MODEL_NATIVE_MANDATORY_FAMILY_FEATURES]"
```

One UTC trading-session clock phases H4 bins on 22/02/06/10/14/18 UTC and D1 at
22:00 UTC; the retired H4 00/04/… and calendar-midnight D1 grids are rejected.
Relevance is learned — no handwritten confluence vote or timeframe weight exists.

## What is implemented

- **The trendline registry was entirely dead and is now alive** (`55148a3b`). A
  same-day regression had replaced the receptive-field lifetime bound with a
  learned value that deleted every line on its own promotion bar. Measured on the
  complete declared tape: 29 of 31 fields were constant or all-NaN on every lane;
  now 0 of 31 are.
- **A full field catalogue exists for the first time**: every field with a
  construction, a liveness verdict and an evidence class, each verdict attacked by
  an independent refuter — 143 were overturned by that attack.
- **First redundancy measurement on this surface**: 15 pairs of 17,391 exceed
  |ρ| 0.999, participation ratio 69.6 of 187 columns, and zero cross-family
  redundancy in the top 25 — the specialist separation is real, not decorative.
- **Price-source parity proven**: canonical tape and the V29 source parquet are
  bit-identical on all four OHLC columns across 476,113 shared rows.
- **The objective is fitted-Q in basis points, not classification** (proven
  from source 2026-08-19). The sole decision loss is masked raw-bps MSE for
  `entry_action_q` and `unified_exit_action`; the decision is the unique argmax
  of `entry_action_q_bps`, not of a calibrated distribution; task weights are
  learned (trainable homoscedastic log-variance); no cross-entropy holds
  decision authority (one masked BCE remains on the `trendline_event`
  auxiliary head). Execute
  `gx1/contracts/entry_model_native_training_objective_v1.py` and
  `entry_model_native_train_recipe_v1.py` for versions, keys and flags — none
  are restated here. The "objective v6 / recipe schema v5 / unweighted CE"
  description carried until 2026-08-19 matched nothing in source.

## What remains empirically unproven or unadmitted

- No dataset, model, calibration, edge, PnL or win-rate. Frequency claims on the
  local signal surface are unproven — that surface has never materialised.
- A decided repair wave covering 86 defective fields (55 repair, 29 retire, 2
  deferred) exists and is **not implemented**.
- **Six-clock squeeze decode mismatch: repaired and refitted 2026-08-18.** The
  2026-08-15 artifacts were absorbing under the runtime decoder on all six
  clocks (reproduced margins, nats: M1 −1.2528, M5 −1.1015, M15 −0.9372, H1
  −1.0991, H4 −1.0279, D1 −0.5260; M1 emitted **one** release in 352,193 TRAIN
  bars). Root cause was **not** the parameters: serve replaced the accumulated
  posterior with a one-hot on the previous argmax, so every state switch had to
  be paid for by a single bar's emission evidence (bounded by the Gaussian
  overlap, ≈2.3 nats) against the fitted persistence penalty log(t11/t10)
  (≈3.5 nats). Refitting under that decoder is impossible — hard-EM with a
  one-step E-step collapses to a single state from the median, memoryless and
  Viterbi starts alike, and the margin stays negative under every reachable
  parameter set. Fit and serve now share one decoder, the causal forward
  filter, which carries the state log-odds and uses no future. `_viterbi_path`
  is deleted, `VOLATILITY_SQUEEZE_FIT_METHOD` is bumped so the old files are
  rejected at load before any data is read, and
  `require_volatility_squeeze_params` proves low-state reachability.
  Refit `VOLATILITY_SQUEEZE_SIXCLOCK_20260818`, measured on real TRAIN bytes:
  release rate 0.0097–0.0167/bar, episodes median 21–36 bars, no constant field
  and no exact-duplicate pair on any clock. **That set was superseded and this
  file named it as current until 2026-08-19**: three non-retired six-clock sets
  now sit under `.../prebuilt/VOLATILITY_SQUEEZE_SIXCLOCK_*` with three
  different `contract_sha256`, and the V31 chains bound
  `..._GEN1f9424_20260818T160532Z`. Read the binding from the run's own V4 cache
  manifest (`volatility_squeeze_artifact_set`), never from a document. No chain
  has carried any of them to GREEN.
- **train==serve is unproven.** Zero `MODEL_NATIVE_SERVE_PARITY` events exist
  under `/home/andre2/GX1_DATA` (measured 2026-08-19): the gate has never
  executed, and it is being repaired now. One divergence is already proven from
  source — the serve ctx-augment HTF block
  (`gx1/execution/v12_ctx_augment_live.py::_atr`) takes a rolling mean of true
  range in float64 while the offline owner (`htf_features._atr` → `wilder_atr`)
  uses Wilder RMA and emits float32, and the serve comment claims they match.
- Whether every static magnitude in the trainer is data-derived is **not
  examined**: the objective contract declares the handwritten-weight flags
  False, but nobody has swept the trainer.

## Coarse history

The system was built and rebuilt four to five times. Each cycle: build, test, the
numbers look good, discover the substrate underneath was broken, rebuild.

- An early chain reached 74.7 bps EV in backtest and was nothing like that live.
- Afterwards most of the feature surface turned out to be air — names without
  substance, or encoding so broken the probes measured the bug rather than the
  market. The eight owners did not agree with each other.
- v9–v19 retired roughly 280 handwritten votes, scorebooks and pre-fused
  composites — all Fibonacci, all candle-pattern votes, the five regime
  composites — replacing them with the raw primitives they were built from.
- The direction edge has been **refuted twice**: the June information-ceiling work
  and an August walk-forward that held in 1 of 5 folds with a −19.48 bps utility
  regression.
- What survived two years is not the code. It is the rules, the gates, and the
  ability to detect that something is wrong before it reaches a model.

## Machine and process safety

Every heavy producer, audit, train or replay enters through
`scripts/gx1_capped_run.sh`: one job at a time, CPU affinity 0-1, 512 MiB swap,
4G for audits and tests, at most 20G for the heavy dataset producers
(`--class producer`) and 20G for the canonical trainer — this file said 10G for
producers until 2026-08-19; `scripts/gx1_capped_run.sh` is the authority. A cap
kill or partial directory is failed evidence.
Deletions under `/home/andre2/GX1_DATA` go through the retention owner only.

## Next implementation sequence

1. Land the repair wave as **one surface generation** — one contract commit, one
   canonical re-materialisation on both clocks, one cache generation, one dataset
   rebuild, one retention pass. Landing it piecemeal pays the same invalidation
   cost N times, which is what the RED V31 chains of 2026-08-18/19 cost. The
   `…_v34` signal contract is in the working tree, uncommitted, alongside the
   parity-gate repair; finish both before spending another chain run.
2. ~~Fix the squeeze decode mismatch and refit the six clocks.~~ Done
   2026-08-18. Chain invocations must pass `--volatility-squeeze-manifest` with
   its matching `--expected-volatility-squeeze-manifest-sha256`. **Resolve that
   pair from the newest artifact set on disk and verify it against the previous
   run's V4 cache manifest — no hash is restated here** (rule 13; the
   `dd051f04…` value this step named was already superseded by the set the V31
   chains actually bound). The 2026-08-15 artifacts fail closed at load on the
   fit-method check regardless.
3. Run the rebuild chain to GREEN and materialise the surface once, on the
   **widened TRAIN window `2021-06-01 → 2026-05-31`** (VAL and TEST unchanged).
   This is not a tuning preference: measured 2026-08-19, the one-year window
   leaves **seven D1 features exactly constant**, which raises
   `[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE]` and prevents the trainer starting
   at all; the five-year window leaves none. 2020 is excluded because its spread
   p90 is 12.46 bps against 1.78–2.82 in every other year. Derivation, the
   per-field measurement and the cost of the change are in
   `docs/TRAIN_WINDOW_WIDENING_20260819.md`; the chain now also fails closed if
   `--history-start` does not cover the D1 receptive field.
4. Re-measure every field against real bytes for a final liveness verdict.
5. Then a **pre-registered cheap test** of whether direction signal exists at all
   on the repaired substrate — question and success criterion written down before
   it runs. The hypothesis has failed on 513 and on 264 columns; it will not be
   rescued by one more column. Do this before spending ten hours on six years of
   data.

## Takeover

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
git worktree list --porcelain
```

Then read `GX1_RULES.md`, `AGENTS.md`, `SYSTEM_MAP.md` and
`docs/DATA_CONTRACT.md`. Do not infer state from old run directories.
