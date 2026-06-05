# Fase-2B rebuild — exact order, inputs, dependencies (learned 2026-06-05, hard-won)

READ THIS before running/resuming the rebuild. It captures the precise ordering, the fail-closed guards,
and the dims map — so we COME HERE AND READ instead of re-scanning the whole chain every session. Each
fact below was discovered by hitting a fail-closed guard and verifying the fix against the cement artifacts.

- **Vedtak:** `fase2b_regime_v4_rebuild_20260605` (REGIME_V4 baked in; flag REMOVED post-cement — no off-switch).
- **Workspace:** `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/` (RUN_MANIFEST.json there).
- **Shell flags for the WHOLE rebuild:** `export GX1_REGIME_V4=1 GX1_TREND_REGIME_FROM_D1=1`.
- **Orchestrator:** `scripts/fase2b_rebuild.sh` (encodes this; idempotent, fail-closed, resumable).

## The 3 "bases" are DIFFERENT LAYERS — ONE truth (do NOT mix fresh + stale)
base28 / base34 / base80 are NOT versions of one base; they are 3 distinct artifacts, each derived from the
SAME clean M5 tape (the one truth). Never run a stale one against fresh ones.
| | what | resolution | purpose | FRESH source this wave |
|---|---|---|---|---|
| **BASE28** | M5 canonical foundation (base features + session + ctx + categoricals) | M5 | source for base34 + base80 | derive from fresh cv3/FULL_PLUS_CTX, NOT `_staging/BASE28_SEED` (stale, ends 03-13, degenerate trend_regime_id) |
| **BASE34** | BASE28 expanded to M1 + ctx16cat6 | **M1** | **serve** source (PrebuiltStateLoader) | rebuild from fresh BASE28; NEVER the stale `MONDAY_WEEK_EXTENSION` base34 |
| **BASE80** | XGB 80-feature set = canonical_v2 + self-attach | M5 | XGB training | **base28 DROPPED** — fresh canonical_v2 (to 05-25) + self-attach (session/multi-TF-v2/cv3-crosses). Build with `--base28-prebuilt NONE`. |
**RULE:** for THIS wave, build everything off the fresh workspace artifacts in `runs/FASE2B_REGIME_V4_20260605/`
+ the pinned clean cv3. The `_staging/*_SEED` + `MONDAY_WEEK_EXTENSION/*` files are STALE — never feed them to a build.
**base80 ↔ base28 root-fix (2026-06-05):** base28 was the ONLY thing capping base80 at 03-13 (its seed ends
2026-03-13) via the inner-join. The 6 base80-contract features base28 uniquely supplied — `session_id` +
`_v1_int_ema_us`/`_v1_int_range_us`/`_v1_int_slope_h1_us`/`_v1_is_EU`/`_v1_is_US` — are DERIVED by the XGB
trainer itself (`train_xgb_universal_multihead_v2.py:120` `_derive_session_context_features`, and `:863-882`
"derived BASE76 missing features"). So `merge_canonical_v2_with_base28_ctx` is now base28-OPTIONAL: pass
`--base28-prebuilt NONE` and base80 follows canonical_v2's full span (to 2026-05-25) with NO 03-13 cap.
Stale base28/base80(03-13) + corrupt-cv3 quarantined reversibly → `runs/_SUPERSEDED_20260605/` (manifest there).

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
   - **A LEAN FULL_PLUS_CTX is OK (don't panic at the col count).** My output = 163 cols vs the cement's 249;
     it carries only the 16 ctx_cont subset + 5 cat + regime + sources. The V10 builder
     (`build_entry_v10_ctx_training_dataset_v3`) **COMPUTES the remaining ~66 ctx_cont itself** (one-truth w/ IQL):
     session one-hots `is_ASIA` (:1220), group-A + dip/struct via `augment_forward_outcome_v2.build_context`
     (:1761-1788), and **fail-closes** if any `ORDERED_CTX_CONT_NAMES_V3` is still missing (:1950-1951). So
     FULL_PLUS_CTX does NOT need all 121 pre-computed — the cement's 249 just had them pre-baked. Verified
     2026-06-05: FULL_PLUS_CTX 395,653 rows, April 4757 (clean), atr_bps + regime cols present.
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

## CORE PATTERN (learned the hard way 2026-06-05): builders are NOT self-contained
Every retrain artifact needs the FULL multi-augmenter chain (multi-TF-v2 `{tf}_*_v2`, session one-hots,
regime, group-A, dip/struct), but the individual builders DON'T all attach it — they assume a prior step did.
The fail-closed guards catch each gap (so nothing poisons silently), but you must attach the missing piece:
- **FULL_PLUS_CTX:** add_ctx_cont now self-attaches `{tf}_*_v2` (fixed 2026-06-05). ✅
- **base80 (`materialize_build_xgb_prebuilt_v3`):** merges v2 + BASE28 but does NOT attach the multi-TF-v2
  (`m15/h1/h4/d1_*_v2`, ~24) NOR the full session set (is_ASIA, minutes_since_session_open, session_change_flag,
  session_tradable, minutes_to_next_session_boundary). XGB sanitizer fail-closes: "Missing 37 features".
  → base80 build must ALSO attach v2_mtf (htf_features.attach_v2_mtf_per_bar_scalars) + the session features.
  Also: BASE28_SEED is stale (ends 2026-03-13) + had no `time` col (reset_index) → base80 caps at 314k rows / 03-13.
- Expect the SAME for the V10/V3/candidate/Exit-IQL dataset builds — attach the missing augmentation per the guard.

**No encoded full-retrain pipeline exists** (cement was ad-hoc over sessions). Completion = days of GPU +
per-artifact augmentation fixes. The honest high-leverage move is to make each builder self-contained
(attach its own augmentation, like add_ctx_cont) so the rebuild becomes reproducible — THEN run.

## VERIFIED build/train recipe — steps 4-7 (the ACTUAL commands, 2026-06-05)
The cement recipe is NOT encoded anywhere — but each cement DATASET dir records it: read
`<dataset>/build.log` + the `*.manifest.json` `build_command` (literal argv) + `DATASET_BUILD_PROOF.json`.
That is the ONE-TRUTH source; reconstruct-from-memory is unreliable (an Explore agent hallucinated 3
trainer script names — ALWAYS `ls` the script before running). Real entrypoints (filesystem-verified):
- XGB train: `gx1.scripts.train_xgb_universal_multihead_v2` (**+ `GX1_XGB_HEAD_CALIBRATE=1 GX1_XGB_CALIBRATOR=isotonic`**
  → writes `CALIBRATION.json` (isotonic_interp, 4 session heads). The V10/V3/candidate builds APPLY this
  calibration to the bridge probs; WITHOUT it the XGB step is incomplete and train≠serve. Deterministic seed=42.)
- V10 transformer train: `gx1.models.entry_v10.entry_v10_ctx_train_v3` (a MODULE under protected `gx1/models/entry_v10/`
  — RUNNING it is fine; only EDITING is gated. `--train`, seed 1337, epochs 10, lr 3e-4, batch 256, seq_len 96.
  COSTFIX cost-matrix is the code DEFAULT (ENTRY_COST_SHORT_TO_LONG=2.00 / FLAT_TO_LONG=1.60 symmetrized).
  ctx dims come FROM the dataset → 121/5 under REGIME_V4, no CLI dim arg.)
- V3 exit train: `gx1.scripts.train_exit_v6_disk_thin`. Entry-IQL build: `gx1.scripts.materialize_build_entry_iql_v2`.
  Candidate batch: `gx1.scripts.materialize_inference_batch_candidates_v3_v1`.
  Exit-IQL per-bar build: `gx1.scripts.materialize_build_exit_iql_per_bar_dataset_v2_m1`.
  (Entry-IQL + Exit-IQL TRAINERS: resolve from their build script / bundle meta — NOT `train_entry_iql_v2`/
  `train_exit_iql_v5` which DO NOT EXIST.)

**V10 dataset build (verified working 2026-06-05, fresh inputs):**
```
GX1_REGIME_V4=1 GX1_TREND_REGIME_FROM_D1=1 GX1_V10_MULTI_TF_V2_CACHE_DIR=<WS>/MULTI_TF_V2_CACHE \
python -m gx1.scripts.build_entry_v10_ctx_training_dataset_v3 \
  --source-parquet-override <WS>/FULL_PLUS_CTX.parquet \
  --xgb-feature-contract-path gx1/xgb/contracts/xgb_input_features_base80_v1.json \
  --xgb-sanitizer-config-path gx1/xgb/contracts/xgb_input_sanitizer_base80_v1.json \
  --xgb_bundle <WS>/xgb_v7 --canonical_v2_parquet <WS>/canonical_features_v2.parquet \
  --tape_root <tape> --output <WS>/v10_dataset/v10_regime_v4_6yr_dataset.parquet \
  --hold-bars 3 --time_split \
  --train_start 2020-11-09T00:00:00Z --train_end 2025-06-30T23:59:59Z \
  --val_start 2025-07-01T00:00:00Z   --val_end   2025-12-31T23:59:59Z \
  --test_start 2026-01-01T00:00:00Z  --test_end  2026-05-25T23:59:59Z
```
Split = train(2020-11-09→2025-06-30) / val(2025-07-01→2025-12-31) / **test = held-out 2026 OOT** (for gates).
NO relabel env — `GX1_V10_RELABEL_RULES` was REMOVED 2026-05-26 (labels are pure outcome-based; the
cement dir name "RELABEL15H24" is legacy, `relabel_veto_rate=0`).

**Self-contained builder fixes made 2026-06-05 (each caught by a fail-closed guard, in order):**
1. base80 `materialize_build_xgb_prebuilt_v3`: base28 OPTIONAL (`--base28-prebuilt NONE`) — see root-fix above.
2. `add_ctx_cont` parquet shape: emit `time` as a plain COLUMN (clean RangeIndex), not a named DatetimeIndex
   (else V10 builder's `reset_index(drop=False)` collides: "cannot insert time, already exists").
3. `add_ctx_cont` features: attach the FULL 9-source mtf_v2 map (was 3 regime-only) + merge cv3 cross-feats
   (smc_premium_state/m5h1_momentum/hour/dow) — else V10 build SOURCE_FEATURES_MISSING (it runs XGB inference
   on FULL_PLUS_CTX → needs ALL 80 base80 features present, except the 6 it derives + is_ASIA).
4. `add_ctx_cont` ctx_cat: under REGIME_V4 source `required_cat` from signal_bridge_v3 (drops trend_regime_id,
   keeps H4_trend_sign_cat) — NOT signal_bridge_v1 EXTENDED[:5] (the opposite) — else CTX_CAT_MISSING_IN_BASE28.

## XGB retrain — VERIFIED CORRECT (B1 OOT gate, 2026-06-05)
After a 79-agent adversarial audit + fixes, the new XGB was OOT-gated on held-out 2026 (23,898 bars, excl the
2026-04 x10-ATR-contamination window; XGB trains only to 2025-12-31). Tool: `runs/FASE2B_REGIME_V4_20260605/
_oot_xgb_directional_compare.py` (reuses XGBMultiheadModel + trainer's compute_triple_barrier_labels — one-truth):
- **NEW dir_acc 0.5077 / logloss 0.9256  vs  CEMENT 0.5018 / 0.9408** → NEW ≥ cement on BOTH (Δ +0.6pp acc, better logloss).
- The in-sample val-margin "collapse" (NEW val_logloss 0.846 vs cement 0.690) was **honest leak-removal, NOT a
  regression**: cement's higher in-sample confidence came from a STALE/leaky mtf_v2 vintage (pre-one-truth-fix);
  NEW uses the corrected serve-faithful `attach_v2_mtf_per_bar_scalars`. Worse in-sample + BETTER OOT = cement
  overfit/leaked, NEW generalizes. **Lesson: never judge a retrain on val/in-sample — OOT decides (project rule).**
- **B2 fixed (train==serve):** base80 builder now BAKES the SHIFTED `_v1_is_EU`/`_v1_is_US` (np.roll by 1, [0]=0)
  matching live serve (`v12_ctx_augment_live.py:165-172`) + cement; trainer log confirms `PRESERVED baked-shifted`.
- Resolver hygiene (rule 8): FG-2 (`--xgb-bundle` made `required` in candidate + V3 builders, no silent stale default).
  STILL OPEN: **FG-1** — `gx1/execution/v12_xgb_live.py:55-57` hardcodes the XGB literal instead of
  `load_decision_artifact("xgb")` (PROTECTED CORE → needs explicit marker; MUST fix before the new chain goes live,
  else live serves the OLD xgb after the contract flips). The 4 transformer/IQL resolvers ARE contract-gated + fail-closed.

## WAVE DECISIONS — dip/top + entry direction (user vedtak 2026-06-05)
Readiness audit found the chain is NOT 100% live end-to-end; user decided how much to bake into THIS cement:
1. **AUG64 = ON** (dip/top into the Exit-IQL POLICY). The V3 exit-transformer already SEES the 64 (36 dip/struct +
   24 group-A + 4 vol), but the IQL policy-head is flag-gated default-OFF. Set `GX1_EXIT_AUGMENT_64=1` for the
   Exit-IQL build/train (and ensure the V3 build emits them via EXIT_IO_V8). Contract-extending → OOT-gate vs cement.
2. **Entry-IQL reward: DROP R_WAIT_OPP_K96_LAM50** — user: "LAM50 var elendig, 99% skip, tok ingenting." Go
   AGGRESSIVE + SYMMETRIC + regime/TF-aware. Candidate: R_V10_PQ_COND_K96 (proven 3.94x takes / 92.99% win, the
   aggressive one rolled back only for a since-fixed CLUSTER1 bug) and/or a LOW-λ per-side R_WAIT_OPP_SYM. Pick at
   the Entry-IQL stage by OOT + the gate below.
3. **HARD ENTRY-DIRECTION GATE (user, emphatic):** the entry must NOT blindly follow the d1 regime. When the
   actionable TFs contradict d1 — m5+m15 DOWN while d1 UP — the model must call SHORT (don't buy the top of a daily
   uptrend). This is LEARNED from `regime_divergence_flag_v3` + per-TF regime classes (now in V10's 121-ctx), NOT a
   hardcoded rule (all-smart-AI). Gate: on a held-out slice where m5&m15 regime=down & d1 regime=up, require the
   retrained V10/Entry-IQL to short (or at minimum NOT go long) at a materially higher rate than cement; this is the
   "short-in-uptrend stress-test gate" — a CEMENT BLOCKER for this wave.
4. **portfolio_parity_B9 = ON** (`GX1_PORTFOLIO_PARITY_B9=1` at forward-outcome regen) — fixes the candidate-density
   train/serve skew (~12-16 train vs ~0-1 live).

## Readiness-audit blockers (2026-06-05) — status
- **[FIXED]** Exit-IQL V8 scorer: `score_v3_v8_on_per_bar_v1.py` SUPPORTED_CONTRACTS now includes EXIT_IO_V8 (171).
- **[OPEN, fix at Entry-IQL stage]** forward-outcome strips OHLC/time → the 36 dip/struct can't attach. Fix:
  `materialize_build_candidate_forward_outcome_dataset_v1.py` carry high/low/close+time OR the 36+24 from the merged
  candidate parquet (inference_batch attaches them @605-607). Plus carry the PLUS5 `_canon_v1` cols in the join.
- **[OPEN, go-live]** FG-1 protected XGB live resolver (needs marker before contract flip).

> Maintenance: when a later step reveals a new input/order/guard, ADD it here the same session (per AGENTS.md rule).
