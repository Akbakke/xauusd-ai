# GX1 model-native data contract

## Scope

GX1 trades XAUUSD only. Large tapes, derived datasets and model artifacts live
under `/home/andre2/GX1_DATA`; the repository contains code and contracts.
Every run consumes explicit immutable absolute paths. Directory conventions
are organizational only and never artifact-selection authority.

## Source bars

The raw market source is complete OANDA `XAU_USD` bid/ask/mid candle data in
UTC. M1 is retained for exit and live-like fill/path reconstruction. M5 is the
Entry decision grid. M15/H1/H4/D1 state must be derived leak-safely using only
bars complete at the decision timestamp.

Required price columns are `open`, `high`, `low`, `close`, `volume`, the four
`bid_*` OHLC columns and the four `ask_*` OHLC columns. Missing bid/ask values,
non-finite prices, duplicates, non-monotonic timestamps or unexplained grid
gaps fail. Mid-only substitution is forbidden.

## Entry split artifact

Each train/validation/test split has an exact immutable manifest with:

- canonical absolute data and source-tape paths;
- file sizes, content hashes and filesystem identity;
- chronological range, row count and leakage/embargo declarations;
- exact model-native signal contract and ordered field hash;
- exact 513 signal, 142 continuous-context and 5 categorical-context names;
- exact target schema and feature/specialist ownership;
- source revision and builder identity.

Splits may not overlap. Calibration and threshold-free evaluation operate only
on their declared split. Test data cannot influence field selection,
normalization, recipe selection, calibration or stopping.

## TRAIN-only state and common history

`model_native_state_contract_v3` is the only accepted rank/history contract.
It requires the exact `model_native_train_rank_reference_v3` payload, whose
NPZ and sidecar both bind the explicit rebuild `--vedtak`.
ATR and observed bid/ask spread are derived causally from raw prices. Their
quintile ranks use one immutable TRAIN-only ECDF whose fit begins at
`TRAIN_START` and ends exactly at `TRAIN_END`; validation, test and serving
reuse it without fitting or updating it.

The rank NPZ contains exactly seven keys: `schema_version`,
`explicit_vedtak_id`, `fit_start_ns`, `fit_end_ns`, `fit_row_count`,
`atr_bps_sorted` and `spread_bps_sorted`. Per-row timestamps, categories and
pinned ATR state are forbidden. Its sidecar binds source and artifact hashes,
declares `fit_scope=train_only`, and proves that no validation or test rows are
stored.

The deterministic feature-ranking JSON that selects the final 174 specialist
fields and the derived seq513 signal manifest are explicit immutable inputs.
Preflight, rebuild wrapper and dataset builder all revalidate their nested
lineage, explicit vedtak, source hash and exact TRAIN start/end against the
requested build. A directory glob, mtime or lexical "latest" result is not an
artifact identity. Any mismatch invalidates the entire attempt; partial files
and non-terminal chain-status records cannot be promoted or resumed as proof.

TRAIN, validation, test and serving all compute features from the same explicit
`feature_history_start_utc`; split-local rolling-state resets are forbidden.
Group-A/dip-structure uses the same implementation offline and live. Only a
contiguous causal warmup prefix may be unavailable and trimmed. A later gap,
non-finite feature, fewer than 96 clean pre-emission history rows, or a stale/v1
contract fails closed. The trim contract also covers every REGIME_V4 source and
derived field, including long-lookback HTF availability, the 288-M5 D1 ROC and
the first observable D1 regime transition.

## Input tensors

The accepted Entry tensor contract is sequence length 96 with 513 genuine
ordered signal fields, a 513-field snapshot, 142 continuous context fields and
5 categorical context fields. The five timeframes are M5/M15/H1/H4/D1.
The 479-field specialist extension is generated inline from that split's common
causal history. It consists of the exact 305 code-owned outputs from all ten
registered causal layers plus 174 eligible fields from deterministic
TRAIN-only ranking. A separately materialized sample-parquet extension is
forbidden.

The seven retired XGB/neutral bridge fields are forbidden. There is no zero-
fill, median-fill, compatibility dimension, optional context or synthetic
decision surface. Missing or non-finite values fail before inference.

## Targets

Targets cover the public `LONG/SHORT/FLAT` decision and every advertised
supporting head: MTF direction, path/MFE/tradability/bad-path/clean-edge/
survival evidence, utility, hierarchy, side validity, trendline rail,
timing/tail/volatility, TF agreement and position size. Target timestamps and
future horizons must be auditable and strictly after the input cutoff.

The position-size target is exactly `sigmoid((MFE-MAE)/(2*ATR_bps))`, where MAE
is a non-negative adverse magnitude; `FLAT` is neutral during training and
executes zero units. Any label-horizon TEST utility/exposure/drawdown result is
diagnostic only; no fresh accepted current-contract result exists. Paper/live
exposure authority additionally requires
a joint sizing-only replay with the exact adopted active Exit stack and a fresh
post-adoption broker runtime-parity event. Until both exist and pass, capital
adoption is blocked; fixed 1x is only a named historical benchmark and never a
fallback.

## Liveness and identity

Full-input liveness evaluates every 513+142+5 field on train, validation and
test. Unallowlisted constants, non-finite values, duplicate/reordered fields,
weak categorical support, forbidden fields and declared ATR drift violations
fail. All five timeframe representations must be present, alive and distinct.

Consumers revalidate bound bytes. A copied manifest or report-level `PASS`
without matching content hashes is not evidence.

No dataset build is authorized merely by this document. Rebuild requires the
exact model-native preflight contract and explicit `--vedtak`.
