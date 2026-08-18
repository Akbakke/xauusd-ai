# GX1 XAUUSD handover

Updated 2026-08-18. `scripts/gx1_handover.sh` is the executable status owner and
outranks this file — run it before relying on anything here. `GX1_RULES.md` is
binding scope; `CLAUDE.md` is the process constitution.

This file was 2,842 words on 2026-08-17 and is now roughly a third of that. What
was cut was a chronological log of every chain attempt, seed run and repair. Git
holds it. A handover is a map of where you are, not a diary.

## Current verdict

Launch is `BLOCK`. There is **no dataset, no model, no calibration, no
untouched-TEST result, no PnL and no win-rate proof** on the current contract.
The offline architecture is connected and repaired at contract/source level; the
feature surface has never been materialised end to end.

The evaluation reference is the coin-flip null: **−13.16 bps TRAIN /
−18.58 bps VAL**. Oracle is +17.76, so available skill is +30.91. Any edge claim
is measured against that null, not against zero.

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
within two days. Read them:

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
- **Objective v6 / recipe schema v5**: plain unweighted CE for main/MTF/
  masked-side classification, plain unweighted BCE for hierarchy binary tasks.
  Waves A/B retired direction and hierarchical distribution forcing.

## What remains empirically unproven or unadmitted

- No dataset, model, calibration, edge, PnL or win-rate. Frequency claims on the
  local signal surface are unproven — that surface has never materialised.
- A decided repair wave covering 86 defective fields (55 repair, 29 retire, 2
  deferred) exists and is **not implemented**.
- **Six-clock squeeze artifacts were fitted 2026-08-15 but are not admissible.**
  The high-volatility state is absorbing under the causal runtime decoder on all
  six clocks: the fit decodes globally (Viterbi/hard-EM) while serve decodes one
  step at a time. That is a rule-6 train≠serve defect at the artifact level and
  needs a preflight, a decode fix, then a refit.
- Fixed auxiliary task weights, rank margins and gate regularization remain a
  Wave-C audit; never claim all static objective magnitudes are gone.

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
4G for audits and tests, at most 10G for the heavy dataset producers and 20G for
the canonical trainer. A cap kill or partial directory is failed evidence.
Deletions under `/home/andre2/GX1_DATA` go through the retention owner only.

## Next implementation sequence

1. Implement the decided repair wave as **one surface generation** — one contract
   commit, one canonical re-materialisation on both clocks, one cache generation,
   one dataset rebuild, one retention pass. Landing it piecemeal pays the same
   invalidation cost N times.
2. Fix the squeeze decode mismatch and refit the six clocks.
3. Run the rebuild chain to GREEN and materialise the surface once.
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
