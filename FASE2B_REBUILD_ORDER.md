# Fase-2B rebuild — exact order, inputs, dependencies (learned 2026-06-05, hard-won)

READ THIS before running/resuming the rebuild. It captures the precise ordering, the fail-closed guards,
and the dims map — so we COME HERE AND READ instead of re-scanning the whole chain every session. Each
fact below was discovered by hitting a fail-closed guard and verifying the fix against the cement artifacts.

- **Vedtak:** `fase2b_regime_v4_rebuild_20260605` (REGIME_V4 baked in; flag REMOVED post-cement — no off-switch).
- **Workspace:** `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/` (RUN_MANIFEST.json there).
- **Shell flags for the WHOLE rebuild:** `export GX1_REGIME_V4=1 GX1_TREND_REGIME_FROM_D1=1`.
- **Orchestrator:** `scripts/fase2b_rebuild.sh` (encodes this; idempotent, fail-closed, resumable).

## The x10 truth (data cleanliness)
- raw M1 canonical: CLEAN. M5 canonical tape `data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL`
  (`year=YYYY/part-000.parquet` ×7, 2020-01-01 → 2026-05-25): CLEAN (April 2026 median 4757).
- The x10 deflation (April 2026 ÷10, median ~475) was ONLY in the cv3/v2 PREBUILT OUTPUT (built when the feed
  was momentarily deflated; the feed has since been corrected). **NOT in the source.** → Fix = rebuild from the
  clean tape (NO rescale). Verify with `gx1/io/price_glitch_guard.assert_no_price_scale_glitch` (0 offending = clean).
- ⚠ The LIVE cv3 + `CANONICAL_V2_PREBUILT` are STILL corrupt (475). Use the freshly-rebuilt + re-pinned ones.

## Spans / model range (why ~10 months "disappear")
- Tape + full v2/cv3 start **2020-01-01**. The MODEL RANGE starts **2020-11-09** — the first ~10 months are
  HTF **warmup** (D1/H4 need ~300 days of history). Cement FULL_PLUS_CTX = 2020-11-09 → 2026-05-22.
- Tape ENDS **2026-05-25** (not 06-04). Extending the last ~10 days needs the live collector M5 appended to the
  tape first — a SEPARATE pre-step if we want data to 06-04.

## Build chain (exact commands + what each produces)
1. **canonical_features_v2** — 118 cols, **HAS `atr`**, 2020-01-01+:
   `python -m gx1.scripts.materialize_build_canonical_features_v2 --out-path <WS>/canonical_features_v2.parquet`
   (reads the clean M5 tape via its default `--m5-root`).
2. **cv3** (the SERVE canonical, read by PrebuiltStateLoader) — 113 cols, **DROPS `atr`** → keeps `atr50`/`atr_z`:
   `python -m gx1.scripts.materialize_canonical_v3_augment --input <WS>/canonical_features_v2.parquet --output-dir <WS>/cv3`
   → `xauusd_m5_CANONICAL_V3_2020_2026.parquet`. Verify glitch-guard, then re-pin to `_PINNED_FASE2B_20260605/`.
3. **FULL_PLUS_CTX** (the V10/V3 BUILD input). `add_ctx_cont` takes the **v2** (NOT cv3 — it needs `atr`),
   **trimmed to 2020-11-09**:
   ```
   GX1_REGIME_V4=1 python -m gx1.scripts.add_ctx_cont_columns_to_prebuilt \
     --prebuilt_parquet <WS>/canonical_features_v2_modelrange.parquet  # v2 filtered to time>=2020-11-09
     --output_parquet <WS>/FULL_PLUS_CTX.parquet \
     --ctx-cont-dim 16 --ctx-cat-dim 5 \
     --tape-root <tape> --raw_m5_parquet <tape>/year=*/part-000.parquet
   ```
   **Fail-closed guards (in the order they fire) — meaning → fix:**
   - `CTX_WARMUP_FAIL` (raw M5 must cover ~300d BEFORE prebuilt start) → trim the PREBUILT (v2) to 2020-11-09;
     the full tape (2020-01-01+) as `--raw_m5_parquet` then provides the ~312-day warmup lead.
   - `CTX_ATR_BPS_FAIL` (prebuilt must contain `atr` to derive `atr_bps`) → use the **v2** (has `atr`), NOT cv3.
   - `REGIME_V4 required source columns missing` (12 `{tf}_*_v2`: m15/h1/h4/d1 × {regime_class_id, trend_age_bars_norm,
     ema_stack_aligned}) → these must be ATTACHED before `add_regime_v4_features`. **One-truth fix (2026-06-05):**
     `add_ctx_cont` now SELF-ATTACHES them via `htf_features.attach_v2_mtf_per_bar_scalars`, mirroring the V3 builder
     (`materialize_build_v3_training_dataset_v2.py:337-353`). Before the fix you'd hand-attach; now it's self-contained.
4. **MULTI_TF_V2_CACHE** regen: `python -m gx1.scripts.prebuild_multi_tf_cache_v2` against the clean cv3/tape
   (verify `manifest.last_ts == cutoff`). REQUIRED by the V10/V3 builds (`GX1_V10_MULTI_TF_V2_CACHE_DIR`); the
   `build_context` stale-guard `[MTF_CACHE_STALE]` refuses a cache lagging the build cutoff by > 2 days.
5. **fresh BASE28 seed** → **base34** (`CTX16CAT6`) → **`backfill_base34_raw_m1_ohlcv_v1 --write`** (R12 M1 volume; run
   AS THE LAST base34 step, re-run after any base34 rebuild; idempotent, atomic, .bak).
6. **V10 build** (`build_entry_v10_ctx_training_dataset_v3`, explicit `--canonical_v2_parquet=<FULL_PLUS_CTX>`)
   → **V3 build** (`materialize_build_v3_training_dataset_v2`, io_version=EXIT_IO_V8) → **candidate-batch**
   (`materialize_inference_batch_candidates_v3_v1`, explicit `--prebuilt/--v10-bundle/--v2-cache-dir`)
   → **Exit-IQL per-bar** (`materialize_build_exit_iql_per_bar_dataset_v2_m1`).
7. **Fase-3 retrain** (`--vedtak fase2b_regime_v4_rebuild_20260605` on EACH trainer — all gated): XGB → V10 →
   Entry-IQL → V3 → Exit-IQL → gates (R13 parity-on-replayed-trade + short-in-uptrend + −2000 reduction, held-out 2026)
   → cement on PASS → **REMOVE the REGIME_V4 flag + OFF/105/6 path** (no off-switch).

## Dims map (the recurring "which dim is which" confusion — STOP re-deriving)
- **ENTRY (V10 → Entry-IQL):** ctx_cont **105** (cement, REGIME_V4=0) / **121** (REGIME_V4=1); ctx_cat **6** / **5**;
  `add_ctx_cont` base-subset = **16** (base6 + micro5 + swing5); base34 = `CTX16CAT6`.
- **EXIT (V3 → Exit-IQL):** EXIT_IO **V6=91 / V7=155** (+4 vol +24 group-A +**36 dip/struct**) **/ V8=171** (+16 regime);
  `exit_io_v1_ctx36` = the older 36-ctx exit contract (the `ctx36` in the branch name).
- **REGIME_V4 ON** = **+16 continuous** regime features (ctx_cont 105→121) and **−1 categorical** (the degenerate
  `trend_regime_id` dropped, ctx_cat 6→5). That IS "bake regime in": rich continuous signal, not an on/off category.

## Verified artifact checkpoints (re-pin / verify at each)
- v2: 456,335 rows × 118 cols, April 4757, has `atr`. cv3: 456,335 × 113, glitch-guard PASS, re-pinned (sha c78181db).
- FULL_PLUS_CTX (cement ref): ~249 cols, has `atr_bps` + the 16 regime cols, 2020-11-09 → 2026.

> Maintenance: when a later step reveals a new input/order/guard, ADD it here the same session (per AGENTS.md rule).
