# GX1 XAUUSD handover

Updated 2026-08-20. `scripts/gx1_handover.sh` is the executable status owner and
outranks this file — run it before relying on anything here. `GX1_RULES.md` is
binding scope; `CLAUDE.md` is the process constitution.

This file was 2,842 words on 2026-08-17; the chronological log of every chain
attempt, seed run and repair was cut and git holds it. **Budget: 2,400 words** —
raised from 1,800 on 2026-08-20 because the chain-binding checklist and the split
derivation are load-bearing for anyone resuming, and cutting them would push the
next session into repeating six failed launches. A handover is a map of where you
are, not a diary; if it grows past the budget, cut history, never a checklist.

## Current verdict

Launch is `BLOCK`. There is **no admitted dataset, no model, no calibration, no
untouched-TEST result, no PnL and no win-rate proof**.

This repository is **offline-only**: no change, rebuild, audit or result here
authorizes paper, demo or live trading.

**V34 was deliberately stopped during `dataset-rebuild` on 2026-08-20.**
`V34_20260820T145741Z` under
`/home/andre2/GX1_DATA/data/data/prebuilt/V34_CHAIN_20260820T145741Z` had no
admitted output when it was stopped: the selective evaluator was proven
unreachable and required a contract repair first. Its partial output is invalid
and may not be resumed or consumed. `CHAIN_STATUS.json` is progress telemetry,
not a terminal admission event.

Everything below the chain is unchanged: no model has ever been trained on this
substrate, and `train==serve` has never been proven (see below).

The evaluation reference is the coin-flip null and it is **substrate-specific**.
The −13.16 / −18.58 bps pair carried until 2026-08-19 was measured on the retired
V27 snapshot and does not transfer. The V31 figures null **−1.87 bps**, oracle
**+23.84**, skill **+25.71** are *reported, not re-derived*: no hash-bound
artifact carries them and no coin-flip owner exists in source. **Re-measuring
them on the admitted rebuild is a precondition of the pre-registered test**, not
an afterthought.

- **Source authority**: pair generation
  `53cba4593471be7532b03a165243506b1add8453886b37a01aca7fb7da4668f7`, published
  2026-08-20 under the existing `CANONICAL_V3_BASE28_BUILDER_V7_20260818T153858Z_GENERATIONS`
  root. It exists because the v34 rename made the previous generation unusable:
  `ema20_slope` / `_v1_ema3_ema6_spread_frac` are gone from its `canonical_v3`
  and the `_atr` spellings are present. Its `base28.parquet` is bit-identical to
  the previous generation's — only the derived surface moved. The previous
  generation `1f9424d8…` is still on disk and is the parent of the V31/V32
  chains; do not reclaim either without a hand-built parent-pointer proof
  (rule 9 — the retention owner cannot do it, see below).
- **Retired, no authority**: the V28 (513) and V29J (592) datasets, and every
  V31/V32 chain root. Nothing was ever trained on any of them.
- **Seed variance flips collapse direction.** Single-seed judging is invalid: one
  three-seed run produced no-collapse, FLAT-drift and LONG-collapse from the same
  recipe. Treat >=5-seed agreement as a gate before any edge claim.

## Current feature architecture

The same eight feature owners, one implementation each, run independently at native M5 for
Entry and native M1 for Exit. Entry reads a local M5 sequence plus closed
M15/H1/H4/D1 context; Exit reads a 480-bar M1 sequence plus closed M5/M15/H1/H4/D1, the
frozen Entry-decision token and its causal in-trade path. The architecture is in
`SYSTEM_MAP.md`; it is not repeated here.

**No width or schema version is restated in this file** (rule 4/13). The counts it
used to carry were stale by 88 fields within two days, and the surface moved again
on 2026-08-19/20. Read them:

```bash
.venv/bin/python -c "import gx1.contracts.entry_model_native_signal_v1 as s, gx1.features.htf_features as h; \
  print('signal', s.MODEL_NATIVE_SIGNAL_DIM, s.MODEL_NATIVE_SIGNAL_SCHEMA_VERSION); \
  print('ctx', s.MODEL_NATIVE_CTX_CONT_DIM, s.MODEL_NATIVE_CTX_CAT_DIM); \
  print('mandatory', s.MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT); \
  print('candidates', s.MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT); \
  print('per_tf', h.MULTI_TF_FEATURE_COUNT_V4, h.HTF_V4_MATRIX_CONTRACT); \
  print('cache', h.HTF_V4_CACHE_SCHEMA_VERSION); \
  [print(' fam', n, len(f)) for n, f in s.MODEL_NATIVE_MANDATORY_FAMILY_FEATURES]"
```

One UTC trading-session clock phases H4 bins on 22/02/06/10/14/18 UTC and D1 at
22:00 UTC. Relevance is learned — no handwritten confluence vote or timeframe
weight exists.

## What is implemented

- **The v34 surface generation** (2026-08-20, `b11ec2b2` — the commit message
  carries every measurement; not repeated here). Eleven fidelity repairs on the
  complete declared tape. The four worth knowing as *classes*, because they
  recur: an exact duplicate the contract had already declared retired while the
  producer kept emitting it (`wick_asym`, with a placeholder 0.0 for an undefined
  0/0); a field that could not express its own name (`local_ema50_slope_bps` was
  exactly `(2/49)·local_price_vs_ema50_bps`, sign agreement 100.000000%); a
  denominator that was a **direction leak** rather than a rescale
  (`ctx_cont.atr_bps` over the bar midpoint — `close/mid` is a monotone
  re-expression of intrabar close position, Spearman 0.917, against a field the
  same vector already carries); and a field whose name named the wrong quantity
  (`spread_intrabar_range_bps` correlated 0.926 with the **bar range**, 0.494 with
  the spread). Nine volatility-coupled fields went from IQR ratio 1.28–1.94 to
  0.966–1.064.
- **Three chain blockers closed.** Each was a restated literal with no owner, and
  each had green tests on both sides: the trainer wrapper hardcoded contract mode
  v18 / width 279 against the owner's v20 / 238; the chain emitted chain-status v9
  while the readiness gate required v7, with one suite pinning each; the parity
  gate passed `bundle_dir=None` into a non-optional parameter. The tests that
  pinned those literals now assert **ownership** — pinning a hardcoded value
  protects it from removal and does nothing to protect it from being wrong.
- **Two crash classes removed.** The normalization fit *inferred* binaryness from
  the sample, so a ternary field observing only {0,+1} was stamped binary and a
  serve-time −1 raised `BINARY_VALUE_INVALID`: no Entry action at all, not even
  FLAT, in a daily downtrend. Declaring the domain would **not** have fixed it —
  `nn.Embedding(len(domain))` is indexed with the raw `.long()` value, so a
  declared `(-1,0,1)` silently indexes the table from the end. The inferred branch
  is removed and the load path now rejects any surface carrying a nonzero
  `binary_mask`. Separately, the chain checked only 96 M5 rows of warmup and never
  the 252-bar daily receptive field.
- **The objective is fitted-Q in basis points, not classification** (proven from
  source 2026-08-19). Sole decision loss is masked raw-bps MSE; the decision is
  the unique argmax of `entry_action_q_bps`, not of a calibrated distribution;
  task weights are learned. No cross-entropy holds decision authority — one masked
  BCE survives on the `trendline_event` auxiliary head. Execute the objective and
  recipe owners for versions and flags. The "objective v6 / recipe v5 / unweighted
  CE" description carried until 2026-08-19 matched nothing in source and had been
  quoted back to the operator as authority.
- **The trendline registry was entirely dead and is now alive** (`55148a3b`):
  29 of 31 fields were constant or all-NaN on every lane, now 0 of 31. A full
  field catalogue exists, every verdict attacked by an independent refuter — 143
  were overturned by that attack.

## What remains empirically unproven or unadmitted

- No dataset, model, calibration, edge, PnL or win-rate. Frequency claims on the
  local signal surface are unproven — that surface has never materialised.
- **Known and deliberately left un-repaired**, recorded so a later FAIL cannot be
  blamed on them retroactively: `mtf_level_bars_since_break` is an exact
  duplicate of `|…_signed|` on all six lanes; the level registry breaks with **no
  confirmation band** (`if close > centre`) and fires on ~19% of bars on every
  clock from M1 to D1 — a property of `SWING_LOOKBACK=3`, not of the market, and
  the rule systematically deletes the levels nearest to price;
  `volatility.squeeze_active` occupies ~87% of bars with no `var0<var1` guard in
  the admission gate; and `level_recurrence_threshold_atr` spans four orders of
  magnitude across seven lanes, with the D1 value fitted on 14 observations.
  The V34 unit repair also made **49 columns bitwise identical** between the
  local M5 surface and the per-TF M5 lane. This is no longer invisible: the
  pre-build cross-surface full scan hashes every active Entry-M5 and Exit-M1
  input against its actually routed MTF last-closed values. Entry excludes M5
  from its MTF route, so those 49 pairs are reported as inactive physical
  overlap; any undeclared duplicate on an active route fails closed. No fresh
  complete run has yet supplied the new report, so this is a source contract,
  not an empirical PASS.
- **Six-clock squeeze**: the 2026-08-15 artifacts were absorbing under the
  runtime decoder on all six clocks — M1 emitted **one** release in 352,193 TRAIN
  bars. The cause was the decoder, not the parameters; fit and serve now share one
  causal forward filter and the old files fail closed at load. The artifacts are
  **TRAIN-window-bound and pair-bound**, so they refit whenever either moves; that
  is why five refits happened on 2026-08-19/20. Never name a squeeze path or hash
  in a document — this file did, and pointed at a superseded set while the chains
  bound a different one. Read the binding from the run's own V4 cache manifest.
  Before admitting a refit, check what the gate does **not**:
  `variances[low] < variances[high]` on every clock, and that the high state is
  not absorbing. The current set passes both, with 1–2 orders of margin and
  high-state runs of 19–26 bars.
- **train==serve is unproven and stays unproven.** Zero
  `MODEL_NATIVE_SERVE_PARITY` events exist (measured 2026-08-19/20). The gate's
  `bundle_dir=None` defect is fixed, so it can now reach a verdict — but there is
  no bundle and no prediction event for it to have a verdict *about*, so rule 6
  belongs in "not examined", never in "proven consistent". The three known
  source divergences are now closed in code, but not yet empirically admitted:
  (a) live no longer overwrites the canonical Wilder ATR with a partial-window
  SMA; (b) the three long-lookback HTF `ctx_cont` fields now delegate to the
  canonical V4 scalar owner and its last-closed projection, rather than a
  private SMA/`ewm`/epsilon formula; and (c) the live Entry MTF builder casts
  OHLCV to the cache's float32 convention before it calls the shared V4 owner.
  The next fresh bundle must still produce a real parity event over these exact
  bound bytes; source agreement is a repair, not evidence of a served model.
- Whether every static magnitude in the trainer is data-derived is **not
  examined**: the objective contract declares the handwritten-weight flags
  False, but nobody has swept the trainer.

## Coarse history

Built and rebuilt four to five times. Each cycle: build, test, the numbers look
good, discover the substrate underneath was broken, rebuild.

- An early chain reached 74.7 bps EV in backtest and was nothing like that live.
- Most of the feature surface then turned out to be air — names without substance,
  or encoding so broken the probes measured the bug rather than the market.
- v9–v19 retired roughly 280 handwritten votes, scorebooks and pre-fused
  composites, replacing them with the raw primitives they were built from.
- The direction edge has been **refuted four times**: the June information-ceiling
  work; an August walk-forward that held in 1 of 5 folds with a −19.48 bps utility
  regression; a GBM on 2026-08-19 where 0 of 5 folds beat the coin flip and OOS
  log-loss was worse than a constant prior; and a horizon sweep where no horizon
  cleared its own floor. **All four measured average accuracy over all bars** —
  see the pre-registered test for why that question nearly guarantees "no".
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

## What will stop you when you run a chain

Six chain attempts on 2026-08-19/20 failed before any compute. Every one was a
binding, not a computation, and every one failed closed in seconds. Check all of
these **before** launching, not after:

1. **Squeeze TRAIN window** must equal the chain's `--train-start/--train-end`.
   Move TRAIN, refit the six clocks. No exception.
2. **Squeeze pair binding** must equal the pair you pass — and the check compares
   the recorded `pair_manifest_artifact` **path**, resolved, not just its hash.
   Fit the squeeze against the *generation-local* `PAIR_MANIFEST.json`, never a
   copy you placed elsewhere, even when the bytes are identical.
3. **Pair manifest** passed to the chain must be the generation-local one.
4. **Worktree must be clean.** The chain binds HEAD; commit first.
5. **Event root must be empty.** A part-built root is never resumed (rule 7).
6. **The pair's `canonical_v3` must carry the current field names.** A base-block
   rename invalidates the pair. Check with `pyarrow` before spending two hours.

The runnable pre-flight is: replicate the gate's own comparison, field for field,
from its source. Verifying the three fields you assumed it checks is how attempt
five failed after attempt four's "ALL BINDINGS OK".

`--registry-fit-train-end` defaults to `--train-end`: one origin. Do not pass it.
`--vedtak` is **inherited** from the native tapes' `explicit_vedtak_id`, never
chosen. A canonical-pair rebuild needs `GX1_V10_MULTI_TF_V4_CACHE_DIR` pointing
at a cache whose manifest carries frozen v29 registry constants — it has no
default by design; the env propagates because `gx1_capped_run.sh` uses
`systemd-run --scope`.

## Next implementation sequence

1. ~~Land the repair wave as one surface generation.~~ Done 2026-08-20,
   `b11ec2b2` (v34 surface: fidelity repairs, three chain blockers, doc truth
   pass) and `e69ab0fb` (sealed-JSON bound derived from the tape).
2. ~~Rebuild the canonical pair on the v34 owners.~~ Done, `53cba459…`.
3. **Start one fresh successor chain; never resume V34.**
   `V34_20260820T145741Z` was stopped before admission and is terminally
   invalid. The successor must use a new event root and run the full
   cross-surface active-input audit now bound into the dataset proof; it may not
   inherit a partial V34 cache, ranking, manifest or split.
4. **Split, and why it is what it is.** TRAIN `2021-06-01 → 2025-05-31` (4y),
   VAL `2025-06-01 → 2026-06-30` (13 months), TEST `2026-07-01 → 2026-08-04T07:50`.
   Four years is a floor, not a preference: below two years the normalization fit
   raises `[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE]` on seven constant D1 fields
   and the trainer cannot start. 2020 is excluded — spread p90 12.46 bps against
   1.78–2.82 everywhere else. **VAL is 13 months because 1 month could not answer
   the question**: 30 days is ~480 independent label windows, 1σ ≈ 2.3pp, against
   an effect size of ~1.6pp. Thirteen months gives ~6,200 windows, 1σ ≈ 0.64pp.
   VAL also spans the 2025–2026 volatility expansion (median M5 bar range ~2.5 →
   ~5.1 USD) by design, so it tests regime transfer, not just fit.
   Derivations: `docs/TRAIN_WINDOW_WIDENING_20260819.md`.
5. Re-measure every field against real bytes for a final liveness verdict.
6. **Run the pre-registered test in
   `docs/PREREGISTERED_DIRECTION_TEST_20260820.md`.** It was
   written before the dataset existed and must not be edited after seeing a
   number. Its central correction: all four previous refutations measured
   *average accuracy over all bars*, which is nearly guaranteed to answer "no"
   whether or not an edge exists — a model that abstains on 92% of bars and has
   real edge on the remaining 8% is invisible in that average. The test asks for
   a selective-edge curve against a re-derived coin-flip null and an
   autocorrelation-preserving (circular-shift) floor, with the decision rule
   fixed in advance.

## Takeover

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
git worktree list --porcelain
```

Then read `GX1_RULES.md`, `AGENTS.md`, `SYSTEM_MAP.md` and
`docs/DATA_CONTRACT.md`. Do not infer state from old run directories.
