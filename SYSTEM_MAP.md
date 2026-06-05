# GX1 SYSTEM MAP — read this BEFORE tracing the chain or any train/serve question

**Purpose:** the ONE place that answers "how is this wired again?" so we stop spending a whole
session re-deriving the same overview. If the answer is here, trust it (each fact carries a
`file:line` you can verify in seconds). If it's NOT here and you had to derive it, **add it here in
the same session** (see "Maintenance rule" at the bottom). This file is version-controlled in the
repo so it travels with the code and is the same truth from `/home/andre2` or from inside the repo.

Scope = the **live COSTFIX chain** (2026-Q2). XAUUSD only. Dates are absolute. Paths are real and
checked on 2026-06-05 unless marked `(verify)`.

---

## 1. The chain (data flow) — what feeds what, at which resolution

```
                      ┌──────── ENTRY (per M5 bar) ────────┐
 M5 canonical ─► XGB v7 base80 ─► 7-dim signal bridge ─► V10 entry transformer ─► Entry-IQL
 (multi-TF)       (M5, NOT per-M1)   p_long/p_short/p_flat/    (per-M5, multi-TF       (3-action:
                                     p_hat/margin/uncertainty/  M5/M15/H1/H4/D1)        SKIP / TAKE_LONG /
                                     entropy                                            TAKE_SHORT — NO WAIT)

                      └──────── EXIT (per M1 bar, in-trade) ────────┐
 latest M5 XGB bridge ─asof-fill─► each M1 bar (held 5 M1 bars; m5_phase_0..4 = staleness)
   + V10 entry-snapshot (frozen at entry) + trade-state (mfe/mae/dd/giveback/...)
   ─► V3 exit transformer (per-M1, 512-bar window + same MTF×5) ─► Exit-IQL (+ MFE-giveback overlay)
```

- **XGB stays M5.** It is NOT recomputed per M1. Exit gets M1 resolution on **price + trade-state**;
  the XGB directional context refreshes at M5 with `m5_phase` encoding staleness. NEVER coarsen the
  exit's M1 price grid to M5 (a PreToolUse hook hard-blocks resample/ffill in exit files).
- **The 7-dim signal bridge** (`p_long,p_short,p_flat,p_hat,margin,uncertainty,entropy`) is the
  SACRED contract feeding BOTH stages. Retrain XGB to fit V10/V3, never refactor V10/V3 to fit XGB.

## 2. Perception vs policy — the part people (and I) keep getting wrong

- **Transformer = the eye.** V10 (entry) / V3 (exit) ingest the multi-TF *sequences* and emit a small
  set of **heads** (direction logits + aux heads like path_quality, mfe_first_n, bad_path, ...).
- **IQL = the brain.** Entry-IQL / Exit-IQL are **flat MLPs** over a vector = `[transformer heads]
  + ctx + trade/portfolio state`. They do **NOT** see the raw TF sequences — only the distilled heads.
- **No WAIT action.** "Wait for the dip" is NOT an action; it is the SKIP reward shaping
  (`R_WAIT_OPP_K96_LAM50` family). Final entry actions are SKIP / TAKE_LONG / TAKE_SHORT.
- Q-values + MAE/MFE/giveback are *really journaled* per bar (no shadow-Q).

## 3. The 19-col V3 trade-state overlay (V4/R13) — ONE TRUTH

The V3 in-trade window has 19 trade-state slots **overlaid** (right-aligned) onto the last
`min(n,512)` bars of the feature window. The 19-col math lives in EXACTLY one place:

- **One truth:** [gx1/features/trade_overlay.py](gx1/features/trade_overlay.py) → `compute_trade_overlay(peak, trough, cur_pnl, atr_bps, entry_snap) -> (n,19) float32`.
- **Build calls it:** [materialize_build_v3_training_dataset_v2.py:480](gx1/scripts/materialize_build_v3_training_dataset_v2.py#L480).
- **Serve calls it:** [v12_trade_state.py `build_v3_overlay`](gx1/execution/v12_trade_state.py) (per-bar intrabar history recorded in `update_bar`).
- **Consumer maps by name:** [v12_v3_live.py `TRADE_STATE_FEATURE_NAMES`](gx1/execution/v12_v3_live.py#L95) — **must equal** `OVERLAY_COL_NAMES` in order (asserted-equal 2026-06-05). Overlay is written to `mat[-n_overlay:, col_idx]`.
- **dtype:** output float32 (the contract); **intermediates float64** in the helper (rounds once at the
  end → more accurate than the pre-V4 float32-stepwise math; ≤1-ULP vs the old inline, retrain absorbs it).

## 4. Parity conventions (the formulas that MUST match build↔serve)

Per-M1 signals come from [compute_per_bar_signals](gx1/scripts/materialize_build_exit_iql_per_bar_dataset_v1.py#L179) (the builder one-truth). Serve mirrors them in `v12_trade_state._intrabar_excursion` / `_pnl_bps`.

| quantity | LONG | SHORT |
|---|---|---|
| entry price | `ask_open[0]` (= serve `entry_ask`) | `bid_open[0]` (= serve `entry_bid`) |
| cur_pnl bps | `(bid_close-entry)/entry*1e4` | `(entry-ask_close)/entry*1e4` |
| peak (favorable) | `(bid_high-entry)/entry*1e4` | `(entry-ask_low)/entry*1e4` |
| trough (adverse) | `(bid_low-entry)/entry*1e4` | `(entry-ask_high)/entry*1e4` |
| atr_bps | `(ask_high-bid_low)/mid*1e4`, `mid=(ask_close+bid_close)/2` | same |

Entry-snapshot (cols 0-4, frozen at entry, from the **V10 direction softmax** — see candidate-gen
[materialize_inference_batch_candidates_v3_v1.py:397-404](gx1/scripts/materialize_inference_batch_candidates_v3_v1.py#L397)):
- `direction_probs = softmax(V10 direction_logits)` in order **[long, short, flat]** (serve: [v12_v10_live.py:394](gx1/execution/v12_v10_live.py#L394)).
- `p_hat = max(3)`, `uncertainty = 1 - p_hat`, `entropy = Shannon natural-log of the 3` (`_compute_entropy_at_entry` == serve `_shannon_entropy`).
- **`margin = top1 - top2` of the 3 probs (sorted[-1]-sorted[-2]). NOT `abs(p_long-p_short)`** —
  that was a serve bug fixed 2026-06-05 in `build_v3_overlay` AND `build_v10_entry_snapshot_features`.

## 5. Live M1 source + the two ATRs

- **Collector parquet:** `/home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_{YYYYMMDD}.parquet`,
  cols `time, open/high/low/close/volume, bid_*, ask_*` (full spread-side OHLC). Written every 60s by
  `v12_oanda_data_collector` (systemd `gx1-collector.service`).
- **Per-M1 atr** = `(ask_high-bid_low)/mid*1e4` (~3-7 bps) ← from the collector parquet via
  [v12_pipeline `_refresh_m1_bar`](gx1/execution/v12_pipeline.py) (V4: one source for BOTH the overlay's
  intrabar peak/trough/atr AND `current_atr_bps_v1`).
- **M5 atr** = `cv3_row["atr_bps"]` (ATR14, ~10-50 bps) → stored as `trade.last_atr_bps`, used ONLY for
  the per-bar journal field + `from_dict` backfill. The OVERLAY no longer reads it (V4).
- The V3-producer daemon ([v12_canonical_incremental.py:347](gx1/execution/v12_canonical_incremental.py#L347))
  carries raw M1 `open/high/low/close/volume` (MID) onto each base34 row — feeds R12 (M1-native volume)
  and is rebuilt for ALL rows at the Fase-2B rebuild. (Intrabar MFE uses spread-side from the collector,
  NOT base34's mid high/low.)

## 6. Prebuilt loading (exit serve)

[v12_state_from_prebuilt.py `PrebuiltStateLoader.load`](gx1/execution/v12_state_from_prebuilt.py) =
**canonical_v3 + BASE28 split** (BASE28 path resolved FRESH from `BASE28_CANONICAL/CURRENT_MANIFEST.json`),
then 5 augmenters in order: volume → v2_mtf_scalars → group_a+dip_struct → v1_legacy → regime_v4
(REGIME_V4 gated). Async background refresh swaps `_cv3/_base28/_last_ts` atomically.
**JOINED single-file fast-path was REMOVED 2026-06-05** (it took unguarded `.exists()` precedence with no
freshness check → stale-poison risk).

## 7. Exit decision flow (per M1 bar)

`v12_paper_runner` loop (per minute, per open trade) → [v12_pipeline.make_exit_decision](gx1/execution/v12_pipeline.py#L425):
1. `_refresh_m1_bar(now)` → `trade.update_bar(bid,ask,m1_close, bid_high,bid_low,ask_high,ask_low)`
   (advances PnL/MFE/MAE + records intrabar peak/trough/atr).
2. `_refresh_canonical(now)` (hot-reload prebuilt, fail-closed on staleness > `GX1_MAX_PREBUILT_STALENESS_MIN`).
3. `build_v3_overlay()` → `v3.predict(... trade_overlay=overlay, multi_tf_windows=...)`.
4. `exit_iql.decide_for_trade(trade, cv3_row, v3_v8_out, current_m1_atr_bps_override=...)` → HOLD / EXIT_NOW.
5. MFE-giveback / Strategy-F overlay may override (E-class, OOT-ablate then default OFF post-retrain).

## 8. Key file → responsibility index (live chain)

| file | owns |
|---|---|
| [gx1/execution/v12_pipeline.py](gx1/execution/v12_pipeline.py) | orchestration: entry + exit decisions, M1-bar refresh, canonical refresh |
| [gx1/execution/v12_trade_state.py](gx1/execution/v12_trade_state.py) | `TradeState`: per-bar PnL/MFE/MAE, intrabar history, V3 overlay, entry-snapshot features, persistence |
| [gx1/execution/v12_v10_live.py](gx1/execution/v12_v10_live.py) | V10 entry transformer inference (direction_probs + aux heads) |
| [gx1/execution/v12_v3_live.py](gx1/execution/v12_v3_live.py) | V3 exit transformer inference; overlays the 19 trade-state cols |
| [gx1/execution/v12_entry_iql_live.py](gx1/execution/v12_entry_iql_live.py) | Entry-IQL (3-action) |
| [gx1/execution/v12_exit_iql_live.py](gx1/execution/v12_exit_iql_live.py) | Exit-IQL (HOLD/EXIT_NOW) + MFE-giveback |
| [gx1/execution/v12_state_from_prebuilt.py](gx1/execution/v12_state_from_prebuilt.py) | `PrebuiltStateLoader`: cv3+BASE28 load, 5 augmenters, async refresh |
| [gx1/execution/v12_canonical_incremental.py](gx1/execution/v12_canonical_incremental.py) | daemon: append new M1 → cv3 + base34 (+raw M1 OHLCV), keep prebuilts ≤5min fresh |
| [gx1/execution/v12_paper_runner.py](gx1/execution/v12_paper_runner.py) | live/paper loop: per-minute entry+exit, OANDA fills, journaling |
| [gx1/features/trade_overlay.py](gx1/features/trade_overlay.py) | ONE-TRUTH 19-col overlay math (build==serve) |
| build: [gx1/scripts/materialize_build_v3_training_dataset_v2.py](gx1/scripts/materialize_build_v3_training_dataset_v2.py) | V3 training dataset (per-bar records + overlay) |
| build: [materialize_build_exit_iql_per_bar_dataset_v1.py](gx1/scripts/materialize_build_exit_iql_per_bar_dataset_v1.py) | `compute_per_bar_signals` + Exit-IQL per-bar dataset |
| build: [materialize_inference_batch_candidates_v3_v1.py](gx1/scripts/materialize_inference_batch_candidates_v3_v1.py) | candidate-gen (V10 softmax → 7-dim bridge rows) |

## 9. Builder ↔ serve mirror table (parity-critical pairs)

| feature/quantity | builder (train) | serve | one-truth |
|---|---|---|---|
| 19-col overlay | materialize_build_v3_training_dataset_v2:480 | v12_trade_state.build_v3_overlay | `trade_overlay.compute_trade_overlay` ✅ |
| per-M1 signals | compute_per_bar_signals:179 | v12_trade_state `_intrabar_excursion`/`_pnl_bps` | mirrored formulas (§4) |
| entry-snap margin | candidate_row["margin"]=top1-top2 | build_v3_overlay + build_v10_entry_snapshot_features | top1-top2 ✅ (fixed 06-05) |
| current_atr_bps_v1 | compute_per_bar_signals atr | v12_pipeline._refresh_m1_bar | `(ask_high-bid_low)/mid` |
| M1-native volume | materialize_build_v3_training_dataset_v2:313 | v12_state_from_prebuilt:507 (**R12: serve still M5-ffill — PENDING**) | ⚠️ not yet one-truth |
| REGIME_V4 ctx (16) | add_regime_v4_features (V3 builder) | v12_state_from_prebuilt._augment_cv3_with_regime_v4 (**R10: serve wire PENDING**) | ⚠️ gated, not fully wired |

## 10. Known live-vs-backtest gaps / gotchas (don't re-discover these)

- **Entry-bar cadence + entry-price basis (R13 RUN-gate).** Builder overlay row 0 = the **entry bar**
  (`entry=ask_open[s_t]`), bars_held 0-based. Live runner opens `TradeState.open(entry_bid=bid, entry_ask=ask)`
  (decision-bar **close**) and the first `update_bar` runs the **next** minute → serve is offset ~1 bar
  and uses a different entry-price basis. This is PRE-EXISTING (the old MVP had it too). Final fix is
  verified by the R13 "per-feature parity assert on a replayed trade" (RUN-gated, post-rebuild).
- **float32 vs float64.** Overlay output is float32 (contract); helper intermediates float64. Build vs
  serve differ ≤~5e-6 bps on the 2nd-diff cols (pnl_acc/giveback) — float32-ULP, harmless, retrain absorbs.
- **Two ATRs** (§5) — M5 (journal) vs per-M1 (overlay + state). Don't cross them.
- **REGIME_V4 flag defaults are OPPOSITE** build vs serve: builder/contract default `1` (ctx_cont 121),
  serve/candidate-gen default `0`, launcher PINS `0` (cement = 105/6). Always set `GX1_REGIME_V4` explicitly
  in a run-manifest; never rely on per-script defaults. (R2/R3/R4 — see RETRAIN_PUNCHLIST.)
- **trend_regime_id was DROPPED** (R4, 2026-06-04): ctx_cat 6→5, fully contract-driven. Don't reintroduce.
- **Exit M1 is sacred.** Never coarsen/downsample to M5 to save compute (hook blocks it). Speed via
  vectorization/numba/GPU.

## 11. Flags index (set explicitly in run-manifests; never trust defaults)

| flag | meaning | build/contract | serve/candidate | launcher |
|---|---|---|---|---|
| `GX1_REGIME_V4` | regime-v4 ctx (121/5 vs 105/6) | `1` | `0` | PIN `0` (cement) |
| `GX1_TREND_REGIME_FROM_D1` | D1-sourced trend regime | — | — | set with REGIME_V4 |
| `GX1_PURE_PHASE6` | disable live-only wrappers (live = Phase-6 OOT 1:1); CLUSTER1_RATE_LIMIT stays ON | — | — | `1` for paper-runner |
| `GX1_MAX_PREBUILT_STALENESS_MIN` | fail-closed SKIP if prebuilt older | — | `30` | — |
| `GX1_STRATEGY_F_ENABLED` | Strategy-F exit overlay (E-class) | — | — | OOT-ablate, default OFF post-retrain |
| `--vedtak <id>` | REQUIRED for any retrain (gx1_guards fail-closed) | — | — | — |

## 12. Protected core + the edit marker

Hard-frozen dirs (CLAUDE.md rule 1): `gx1/execution`, `gx1/contracts`, `gx1/exits/contracts`,
`gx1/models/entry_v10`, `gx1/core`. A PreToolUse hook ([.claude/hooks/guard_write.py](.claude/hooks/guard_write.py))
blocks edits unless a **one-shot** marker exists: `touch /home/andre2/src/GX1_ENGINE/.claude/ALLOW_CORE_EDIT`
(consumed after a single write, re-arms). Lifting it is the user's explicit act, per change. The same hook
HARD-BLOCKS M1→M5 coarsening in exit files.

---

## Maintenance rule (this is why we stop re-deriving)

**Before** tracing the chain or answering a train/serve / data-flow / parity question, READ THIS FILE.
**After** you derive any non-obvious fact the map didn't have (a new call site, a formula, a flag, a
gotcha, a moved file), ADD it here in the SAME session, with a `file:line`. Keep it TIGHT — facts +
pointers, not prose; this is a map, not a log (logs live in DECISION_LOG.md / PROJECT_STATE.md).
When code moves, fix the pointer here in the same change. One truth, fail-closed, current.
