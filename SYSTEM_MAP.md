# GX1 SYSTEM MAP — read this BEFORE tracing the chain or any train/serve question

**Purpose:** the ONE place that answers "how is this wired again?" so we stop spending a whole
session re-deriving the same overview. If the answer is here, trust it (each fact carries a
`file:line` you can verify in seconds). If it's NOT here and you had to derive it, **add it here in
the same session** (see "Maintenance rule" at the bottom). Version-controlled in the repo so it's the
same truth from `/home/andre2` or inside the repo. Scope = the live **fase2b/CLEAN cement** (entry
chain `FASE2B_REGIME_V4_20260605`, exit chain `FASE2B_CLEAN_20260608`; cemented 2026-06-08, clean-exit
flipped 2026-06-10), XAUUSD only. Paths/lines checked 2026-06-05 unless marked `(verify)`.

Layout: §1-9 = the chain + parity (read first). §10-15 = stack internals (entry, exit, contracts,
XGB, multi-TF, datasets). §16-18 = gotchas, flags, protected core. Then the maintenance rule.

---

## 1. The chain (data flow) — what feeds what, at which resolution

```
                      ┌──────── ENTRY (per M5 bar) ────────┐
 M5 canonical ─► XGB v7 base80 ─► 7-dim signal bridge ─► V10 entry transformer ─► Entry-IQL
 (multi-TF)       (M5, NOT per-M1)   p_long/p_short/p_flat/    (per-M5, multi-TF       (3-action:
                                     p_hat/margin/uncertainty/  M5/M15/H1/H4/D1)        SKIP / TAKE_LONG_NOW /
                                     entropy                                            TAKE_SHORT_NOW — NO WAIT)

                      └──────── EXIT (per M1 bar, in-trade) ────────┐
 latest M5 XGB bridge ─asof-fill─► each M1 bar (held 5 M1 bars; m5_phase_0..4 = staleness)
   + V10 entry-snapshot (frozen at entry) + trade-state (mfe/mae/dd/giveback/...)
   ─► V3 exit transformer (per-M1, 512-bar window + same MTF×5) ─► Exit-IQL (+ MFE-giveback overlay)
```

- **XGB stays M5.** NOT recomputed per M1. Exit gets M1 resolution on **price + trade-state**; the XGB
  directional context refreshes at M5 with `m5_phase` encoding staleness. NEVER coarsen the exit's M1
  grid to M5 (a PreToolUse hook hard-blocks resample/ffill in exit files).
- **V3 window = EXACTLY 512 M1 bars (8.5h)**, assembled once per M1 bar: canonical from BASE34, XGB
  bridge asof from M5 buckets (`bridge_by_ts` map), 19-col overlay right-aligned onto last `min(n,512)` rows ([v12_v3_live.py:257-347](gx1/execution/v12_v3_live.py#L257)).
- **Entry-IQL `state_v1` (197-dim raw, un-normalized)** is journaled as `entry_iql_state_v1` for online-IQL replay + offline counterfactual variant replay ([v12_pipeline.py:390-438](gx1/execution/v12_pipeline.py#L390)). (Audit 2026-06-08: actual cement+clean = **197**, NOT 192; the adapter is name-driven and asserts `state_dim == len(feature_names)`.)

## 2. Perception vs policy — the part people (and I) keep getting wrong

- **Transformer = the eye.** V10 (entry) / V3 (exit) ingest multi-TF *sequences* and emit small **heads**.
- **IQL = the brain.** Entry-IQL / Exit-IQL are **flat MLPs** over `[transformer heads] + ctx + trade/portfolio
  state` — they do NOT see the raw TF sequences.
- **No WAIT action.** Entry actions = SKIP / TAKE_LONG_NOW / TAKE_SHORT_NOW. "Wait for the dip" = the SKIP
  reward shaping (`R_WAIT_OPP_K96_LAM50` family), not an action. Exit actions = **HOLD=0 / EXIT_NOW=1** only.
- **Entry-IQL K-horizons** = `[12,24,48,96,144,192]` M5 bars (1/2/4/8/12/16h); multi-head Q per (action,K) cell;
  aggregator mean/max/weighted; `K_PRIMARY=96` derives the oracle stratification label ([materialize_build_entry_iql_v2.py:87](gx1/scripts/materialize_build_entry_iql_v2.py#L87), [entry_iql_v2_adapter.py:54-269](gx1/runtime/entry_iql_v2_adapter.py#L54)).
- **Exit-IQL K-horizons** = `[1,4,12,48,144,240]` **M1** bars (1min..4h) — SCALP lookahead; K = window for the
  hold-max-pnl term, NOT exit time. DIFFERENT from entry's K-set ([materialize_build_exit_iql_per_bar_dataset_v1.py:101](gx1/scripts/materialize_build_exit_iql_per_bar_dataset_v1.py#L101)).
- **Advantages (bps, reward scale):** `advantage_over_skip = Q[chosen]-Q[SKIP]`; `advantage_over_realized =
  Q[chosen]-max(Q[long],Q[short])` ([entry_iql_v2_adapter.py:75-78,297](gx1/runtime/entry_iql_v2_adapter.py#L75)). Exit advantage = `q_exit-q_hold`, gated by `exit_margin` (default 0 = argmax).
- **Entry selection = conviction-gate (thr −37.71) + margin²×inverse-ATR sizing + skip-ASIA + DIPFIX(both)**
  (serve-time overlay; the `entry_iql_volbal_20260611` **bundle is UNCHANGED**): OPEN a TAKE when
  `raw_adv = best-TAKE-Q − SKIP-Q` (UN-clipped) `>= GX1_CONVICTION_THR` (live **−37.71**; the open-more-to-−100
  wave was REVERTED — launcher+contract+06-21 memory all −37.71; the `open100` journal-suffix is a stale name;
  was −34.2/top-20% cemented 2026-06-10), side = `argmax(TAKE-Q)`, OVERRIDING the IQL argmax-SKIP; + skip-ASIA.
  Env-gated `GX1_CONVICTION_GATE`/`GX1_SKIP_ASIA`/`GX1_CONVICTION_THR` ([v12_entry_iql_live.py:104](gx1/execution/v12_entry_iql_live.py#L104),
  [v12_paper_runner.py `size_units`](gx1/execution/v12_paper_runner.py); launcher pins `1`/`1`/`−37.71`).
  ONE truth = `PROJECT_STATE_artifacts.json` → `entry_iql.operating_point`.
  - **GATE FIDELITY GOTCHA (2026-06-23):** the offline gate (`gx1_candidate_gate.sh` → `v12_phase6_joint_validation`)
    does **NOT** replicate this selection — `v12_phase1_entry_iql_inference` loads with `min_advantage_bps=0`
    ⇒ emits the **argmax** action (+DIPFIX +margin-floor), phase6 replays `action_label_v1` and uses
    `advantage_over_skip_v1` only for the per-side calibration check, posthoc applies only skip-ASIA. So the
    gate measures the **argmax-IQL+DIPFIX policy, equal-weighted** — NO conviction-gate open-more, NO margin²-sizing.
    For entry overlays that target the raw_adv-ADMITTED marginal-conviction trades (e.g. the **margin-floor** overlay
    `GX1_ENTRY_MARGIN_FLOOR`, `apply_margin_floor_overlay`, commit 0d9e74b0), those trades are argmax-SKIP ⇒ ABSENT
    from the gate ⇒ it FALSE-REFUTES. Faithful instrument = [step1_roundwall_ab.py](gx1/research/step1_roundwall_ab.py)
    (imports live constants, parity-checks vs real `predict()`, conviction-gated base replay + `simulate_portfolio(cap=3)`
    + size-weighting). See `GX1_DATA/MARGIN_FLOOR_GATE_RUNBOOK_20260623.md`.
  - **POSITION SIZING (`SIZING_MODE=both`):** `size = clip( (margin²/REF) × min(ATR_REF/atr, 1), MIN_MULT, MAX_MULT )`
    — a conviction `margin^POW/REF` leg (margin = XGB/V10 top1−top2 = 1−uncertainty, corr 0.36 w/ realized) **×** an
    inverse-ATR `min(14/atr,1)` leg that ONLY down-sizes high-vol bars; final clamp `[0.5, 2.0]`. Live: `POW=2.0`,
    `REF=0.3318`, `ATR_REF=ATR_FLOOR=14`. **FOOTGUN — REF must track POW:** the launcher pins `REF=0.3318` (=full-history
    mean(margin²)) and does NOT reset it when POW changes; flipping only `POW=1.0` leaves REF=0.3318 → sizes ~everything
    to 2× (clipped), NOT gentle margin¹. Mean-preserving per-POW population REF (2026): POW 1.0→0.7046, 1.5→0.6347,
    2.0→0.5859 (2026 mean(margin²)=0.5859 > 0.3318 → live runs ~1.77× avg, the intended "more gas").
    `V12Pipeline.make_entry_decision()` must surface `margin`/`margin_top1_top2` at top level because
    `v12_paper_runner.size_units()` reads the decision dict, not the local Entry-IQL candidate.
  - **TRUE cap-3 per-M1-bar mark-to-market account DD = 564 bps, ret/DD 30.5** (the inverse-ATR leg already de-risks the
    EXTREME-vol tail; worst moment = ≤3 concurrent same-side trades underwater that RECOVER, not lost capital). The "366"
    is a realized-bps proxy; "888" is margin²-only (omits the inverse-ATR leg). Entry-IQL retrain on the toxic LONG
    cluster is OOT-REFUTED (memory `project_gx1_dd_analysis_retrain_refuted_20260614`) — do NOT build it.
  2026-06-11 gate-hoist fix: the gate+overlays now apply on the single-bundle `predict()` path too (they
  were ensemble-branch-only — the flag was a live NO-OP between 2026-06-10 and the fix; [v12_entry_iql_live.py:417-419](gx1/execution/v12_entry_iql_live.py#L417)).
  - SUPERSEDED (pre-2026-06-10, "LAM50/VOLUME-FIRST" *operating point* — NOT the bundle): cement
    `min_advantage_bps = 0.0` (OFF) + runtime adaptive `min_adv = max(1.5, 0.35×ATR_bps)` = IQL argmax-SKIP,
    ~8% take, admit 0.458 < gate floor. Retired by the conviction-gate.

## 3. The 19-col V3 trade-state overlay (V4/R13) — ONE TRUTH

19 trade-state slots **overlaid** (right-aligned) onto the last `min(n,512)` window bars. Math lives in EXACTLY one place:
- **One truth:** [gx1/features/trade_overlay.py](gx1/features/trade_overlay.py) → `compute_trade_overlay(peak,trough,cur_pnl,atr_bps,entry_snap) -> (n,19) float32`.
- **Build:** [materialize_build_v3_training_dataset_v2.py:480](gx1/scripts/materialize_build_v3_training_dataset_v2.py#L480). **Serve:** `v12_trade_state.build_v3_overlay` (intrabar history in `update_bar`).
- **Consumer:** `TRADE_STATE_FEATURE_NAMES` = **indices 7-25** of `V6_FEATURES`; **stable across V6/V7/V8** (prefix-identical) ([v12_v3_live.py:95-103,338](gx1/execution/v12_v3_live.py#L95)). Must equal `OVERLAY_COL_NAMES` (asserted 2026-06-05).
- **dtype:** float32 output (contract); float64 intermediates in the helper (≤1-ULP vs the old inline; retrain absorbs).

## 4. Parity conventions (the formulas that MUST match build↔serve)

Per-M1 signals: [compute_per_bar_signals](gx1/scripts/materialize_build_exit_iql_per_bar_dataset_v1.py#L179) (one-truth); serve mirrors in `v12_trade_state._intrabar_excursion`/`_pnl_bps`.

| quantity | LONG | SHORT |
|---|---|---|
| entry price | `ask_open[0]` (= serve `entry_ask`) | `bid_open[0]` (= serve `entry_bid`) |
| cur_pnl bps | `(bid_close-entry)/entry*1e4` | `(entry-ask_close)/entry*1e4` |
| peak (favorable) | `(bid_high-entry)/entry*1e4` | `(entry-ask_low)/entry*1e4` |
| trough (adverse) | `(bid_low-entry)/entry*1e4` | `(entry-ask_high)/entry*1e4` |
| atr_bps | `(ask_high-bid_low)/mid*1e4`, `mid=(ask_close+bid_close)/2` | same |

Entry-snapshot (cols 0-4, frozen, from **V10 direction softmax** — candidate-gen [...candidates_v3_v1.py:397-404](gx1/scripts/materialize_inference_batch_candidates_v3_v1.py#L397)):
- `direction_probs = softmax(V10 direction_logits)` in order **[long,short,flat]** (serve [v12_v10_live.py:394](gx1/execution/v12_v10_live.py#L394)).
- `p_hat = max(3)`; `uncertainty = 1 - p_hat`; `entropy = Shannon natural-log` (`_compute_entropy_at_entry` == serve `_shannon_entropy`).
- **`margin = top1 - top2` (sorted[-1]-sorted[-2]), NOT `abs(p_long-p_short)`** — serve bug fixed 2026-06-05 in `build_v3_overlay` AND `build_v10_entry_snapshot_features`. (Same formula in the bridge: [signal_bridge_v3.py:62-68](gx1/contracts/signal_bridge_v3.py#L62).)
- **V10 forward-outcome rescore contract:** use `gx1/scripts/rescore_forward_outcome_v10v2_full.py` for new-eyes retrains.
  It must overwrite the base Entry-IQL state columns as well as emit `_v2` lineage columns; `_v2`-only output is
  invalid because `materialize_build_entry_iql_v2.py` reads base names (`p_long`, `margin`, `bad_path_prob`,
  `v10_dip_*`, ...). The legacy partial rescore is disabled under `gx1/scripts/_legacy_disabled/`.
- **V10 `--eval` contract:** eval must use the same manifest-driven runtime bundle loader and multi-TF dataset kwargs
  as train/export. Manual single-TF model construction is invalid for modern V10 bundles with cross-TF/new-head params.
- **ACTIVE Entry/Exit feature-coverage contract (2026-06-26 audit):** active Entry-IQL
  `entry_iql_volbal_20260611/R_WAIT_OPP_K96_LAM50_SYM/FOLD_1` loads 197 features / 189 required; live candidate
  audit had `required_missing=0` and nonzero V10 new-head groups (`v10_dip`, `v10_forecast`, `v10_timing`,
  `v10_tail_risk`, `v10_vol_forecast`). Active Exit-IQL `exit_iql_retrain_clean_20260609/R_NET_REAL/FOLD_1`
  loads 209 features / 202 required; live-loader bar_state audit had `required_missing=0` after PrebuiltStateLoader
  AUG64 augmentation. Regression: `tests/test_iql_adapter_emitter_parity.py` now has PROJECT_STATE-following active
  Entry coverage and an opt-in full live-loader Exit coverage test (`GX1_RUN_LIVE_LOADER_CONTRACT_TESTS=1`).
- **Active Entry-IQL artifact footgun:** `summary_v1.input_forward_outcome_dir_v1` is not by itself the final
  197-dim state contract; some source forward_outcome shards lack required dip/struct cols. Use the bundle adapter
  feature list plus `drift_reference_v1.parquet` for active state-contract tests.

## 5. Live M1 source + the two ATRs

- **Collector parquet:** `/home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_{YYYYMMDD}.parquet`,
  cols `time, open/high/low/close/volume, bid_*, ask_*`. Written by `v12_oanda_data_collector` (systemd `gx1-collector.service`).
  Poll cadence env `GX1_COLLECTOR_POLL_SECONDS` (default **60**; **live = 15** via drop-in `gx1-collector.service.d/poll15.conf`, 2026-06-08) ([v12_oanda_data_collector.py:50-57](gx1/execution/v12_oanda_data_collector.py#L50)). 15s = a closed M1 bar reaches disk within ~15s; this parquet feeds `m1_close` + canonical/feature freshness, **NOT** the live exit price (see §16 lag-truth).
- **Two independent reads of this parquet per exit decision:** (1) the runner reads `ask_close+bid_close` →
  `m1_close` (mid, **NOT cached**) ([v12_paper_runner.py:589-598](gx1/execution/v12_paper_runner.py#L589)); (2) `_refresh_m1_bar` reads intrabar high/low for the overlay peak/trough/atr (**cached per minute**) ([v12_pipeline.py:150-191](gx1/execution/v12_pipeline.py#L150)).
- **Per-M1 atr** `(ask_high-bid_low)/mid*1e4` (~3-7 bps) ← `_refresh_m1_bar` (ONE source for overlay intrabar + `current_atr_bps_v1`).
- **M5 atr** `cv3_row["atr_bps"]` (ATR14, ~10-50 bps) → `trade.last_atr_bps`, used ONLY for journal + `from_dict` backfill. Overlay no longer reads it (V4).
- The V3-producer daemon ([v12_canonical_incremental.py:347](gx1/execution/v12_canonical_incremental.py#L347)) carries raw M1 `o/h/l/c/v` (MID) onto base34 (R12 volume); intrabar MFE uses **spread-side** from the collector, not base34 mid.

## 6. Prebuilt loading (exit serve)

[PrebuiltStateLoader.load](gx1/execution/v12_state_from_prebuilt.py) = **canonical_v3 + BASE28 split**, then 5 augmenters: volume → v2_mtf_scalars → group_a+dip_struct → v1_legacy → regime_v4 (REGIME_V4 gated).
- **BASE28 path resolved FRESH from manifest** every `refresh_if_changed` (rotation-safe; cached path only as read-failure fallback) ([:159-170](gx1/execution/v12_state_from_prebuilt.py#L159)). JOINED single-file fast-path **removed 2026-06-05** (unguarded `.exists()` stale-poison risk).
- **REGIME_V4 source:** `D1_dist_from_ema200_atr` is the ONLY input NOT emitted by `_augment_cv3_with_v2_mtf_scalars` (which emits the 12 `{tf}_*_v2` cols); ffilled from BASE28, **fail-close** if missing ([:403-414](gx1/execution/v12_state_from_prebuilt.py#L403)).
- **Async refresh atomicity:** multi-TF feats built on a LOCAL cv3 and swapped TOGETHER with cv3 in one GIL-atomic block (2026-06-01 fix vs cv3↔mtf divergence) ([:284-312](gx1/execution/v12_state_from_prebuilt.py#L284)). MP-pool failure → sequential same-augmenter fallback (bit-parity); daemon try/catch keeps stale on hiccup (never crashes).
- `_augment_cv3_with_v2_mtf_scalars` (2026-06-04 one-truth) calls `htf_features.attach_v2_mtf_per_bar_scalars` → emits **31** `{tf}_{feat}_v2` XGB cols byte-identically to the V3 builder ([:559-600](gx1/execution/v12_state_from_prebuilt.py#L559)).
- **group_a+dip_struct builds multi-TF IN-MEMORY in live serve (2026-06-08, commit d4c09c6f):** `_augment_cv3_with_group_a_and_dip_struct` builds the V2 mtf bundle from THIS cv3 (`build_multi_tf_per_bar_features_v2`) and passes it to `attach_group_a_dip_struct_ctx_columns(..., multi_tf=…)` ([:450-470](gx1/execution/v12_state_from_prebuilt.py#L450)) — it no longer reads the on-disk `MULTI_TF_V2_CACHE`. So **live serve has ZERO disk-MTF-cache dependency** (mechanism-A windows + v2_mtf_scalars were already in-memory); the disk cache is now BUILD-pipeline-only. `attach_…`'s new `multi_tf` param = None → unchanged disk-cache path for builds ([augment_forward_outcome_v2.py:521-575](gx1/scripts/augment_forward_outcome_v2.py#L521)). train==serve preserved (same builder, causal/asof → bit-identical at every decision ts).

## 7. Exit decision flow (per M1 bar)

`v12_paper_runner` loop → [v12_pipeline.make_exit_decision](gx1/execution/v12_pipeline.py#L444):
1. `_refresh_m1_bar` → `trade.update_bar(bid,ask,m1_close, bid_high,bid_low,ask_high,ask_low)` (advances PnL/MFE/MAE + records intrabar).
2. `_refresh_canonical` (hot-reload; fail-closed if stale > `GX1_MAX_PREBUILT_STALENESS_MIN`).
3. `build_v3_overlay()` → `v3.predict(... trade_overlay, multi_tf_windows)`.
4. `exit_iql.decide_for_trade(trade, cv3_row, v3_v8_out, current_m1_atr_bps_override)` → `(rec, bar_state)`; `action_id` 0=HOLD/1=EXIT_NOW; `bar_state` journaled for distillation ([v12_exit_iql_live.py:468](gx1/execution/v12_exit_iql_live.py#L468)).
- **Trade-state = 18 feats** (`build_trade_state_features`): bars_in_trade, pnl, mfe, mae, bars_since_mfe, dd, bar_return, 3× rolling ret (5/15/60), 2× rolling vol (15/60), pnl_vel, pnl_acc, mfe_decay (cum_peak[t]-cum_peak[t-4]), giveback (clip[-10,10]), giveback_acc (2nd diff), rolling_slope (OLS) ([v12_trade_state.py:270-348](gx1/execution/v12_trade_state.py#L270)).
- **V10 entry-snapshot = 10 feats** frozen at open (6 V10 + 4 v3+ aux tf_agreement/path_quality_std/position_size/hold_horizon; `hold_horizon=-1` sentinel = bundle lacks head) ([v12_trade_state.py:350-387](gx1/execution/v12_trade_state.py#L350)).
- **m5_phase live = `minute%5`** via `compute_m5_phase_onehot` (matches trainer); old canon path = `minute//12` (stale, different formula) ([v12_exit_iql_live.py:384-399](gx1/execution/v12_exit_iql_live.py#L384)).

## 8. Key file → responsibility index (live chain)

| file | owns |
|---|---|
| [v12_pipeline.py](gx1/execution/v12_pipeline.py) | orchestration: entry + exit decisions, M1-bar refresh, canonical refresh |
| [v12_trade_state.py](gx1/execution/v12_trade_state.py) | `TradeState`: PnL/MFE/MAE, intrabar history, V3 overlay, entry-snapshot, persistence |
| [v12_v10_live.py](gx1/execution/v12_v10_live.py) | V10 entry transformer (direction_probs + aux heads) |
| [v12_v3_live.py](gx1/execution/v12_v3_live.py) | V3 exit transformer; overlays 19 trade-state cols; io_version detect |
| [v12_entry_iql_live.py](gx1/execution/v12_entry_iql_live.py) | Entry-IQL (3-action) + 43 V10-head unpack |
| [v12_exit_iql_live.py](gx1/execution/v12_exit_iql_live.py) | Exit-IQL (HOLD/EXIT_NOW) + Strategy-F + distilled-Q swap |
| [v12_state_from_prebuilt.py](gx1/execution/v12_state_from_prebuilt.py) | `PrebuiltStateLoader`: cv3+BASE28 load, 5 augmenters, async refresh |
| [v12_canonical_incremental.py](gx1/execution/v12_canonical_incremental.py) | daemon: append M1 → cv3 + base34 (+raw M1 OHLCV), HTF recompute, ≤5min fresh |
| [v12_xgb_live.py](gx1/execution/v12_xgb_live.py) | XGB base80 inference, session→head routing, contract cross-check, NaN fail-close |
| [v12_paper_runner.py](gx1/execution/v12_paper_runner.py) | live/paper loop: per-minute entry+exit, OANDA fills, safety, journaling |
| [trade_overlay.py](gx1/features/trade_overlay.py) | ONE-TRUTH 19-col overlay math |
| [htf_features.py](gx1/features/htf_features.py) | multi-TF per-bar features (V2), `MULTI_TF_SHIFT`, warmup zero-pad |
| build: [materialize_build_v3_training_dataset_v2.py](gx1/scripts/materialize_build_v3_training_dataset_v2.py) · [..._exit_iql_per_bar_dataset_v1.py](gx1/scripts/materialize_build_exit_iql_per_bar_dataset_v1.py) · [..._entry_iql_v2.py](gx1/scripts/materialize_build_entry_iql_v2.py) · [..._inference_batch_candidates_v3_v1.py](gx1/scripts/materialize_inference_batch_candidates_v3_v1.py) | training datasets + candidate-gen |

## 9. Builder ↔ serve mirror table (parity-critical pairs)

| quantity | builder | serve | status |
|---|---|---|---|
| 19-col overlay | build_v3_dataset_v2:480 | v12_trade_state.build_v3_overlay | ✅ `compute_trade_overlay` |
| per-M1 signals | compute_per_bar_signals:179 | `_intrabar_excursion`/`_pnl_bps` | ✅ mirrored (§4) |
| entry-snap margin | top1-top2 | both serve sites | ✅ top1-top2 (fixed 06-05) |
| current_atr_bps_v1 | compute_per_bar_signals atr | `_refresh_m1_bar` | ✅ `(ask_high-bid_low)/mid` |
| 31 `{tf}_*_v2` cols | V3 builder attach_v2_mtf | `_augment_cv3_with_v2_mtf_scalars` | ✅ byte-identical (06-04) |
| m5_phase | `minute%5` (trainer) | `compute_m5_phase_onehot` | ✅ (old `minute//12` retired) |
| M1-native volume (4 feats, idx 91-94) | build_v3_dataset_v2:313 (M1-native) | v12_v3_live:307-318 (M1-native branch, **present**, one-truth `compute_volume_features`) | ⚠️ **rebuild-gated, NOT a code edit**: serve branch fires only when base34 carries raw `volume`; base34 has it on NEW rows only (incr. daemon), needs full-rebuild to backfill ALL rows. M5-ffill fallback until then. |
| REGIME_V4 ctx (**18**) | add_regime_v4_features | `_augment_cv3_with_regime_v4` | ✓ **serve-wired** (gated GX1_REGIME_V4; launcher pins =1) |

---

## 10. Entry stack internals (V10 → Entry-IQL)

- **V10 multi-TF is mandatory.** V10 (M5-base) requires `seq_m15/h1/h4/d1` (and `seq_m5` iff `enable_multi_tf_m5`);
  `V10.predict()` raises `RuntimeError` if `multi_tf_windows=None`. Default per-TF `seq_len=96`, bundle-metadata-driven ([v12_v10_live.py:358-385](gx1/execution/v12_v10_live.py#L358)). Candidate-gen hard-rejects non-multi-TF V10 ([...candidates_v3_v1.py:637-641](gx1/scripts/materialize_inference_batch_candidates_v3_v1.py#L637)).
- **43 V10 new-head scalars** unpacked into Entry-IQL: dip 18 / forecast 4 / timing 12 / tail_risk 6 / vol_forecast 3;
  missing heads → explicit 0-fill + structured warn per missing-set change (2026-06-02 fix for the all-LONG-bias 0-fill bug) ([v12_entry_iql_live.py:275-310](gx1/execution/v12_entry_iql_live.py#L275)).
- **Entry-IQL state:** cement runtime `state_v1` = **197-dim** (raw, journaled; verified at source 2026-06-08 — ckpt `net.0.weight=(256,197)`, 197 named `feature_names_v1`). The adapter builds the state strictly BY NAME with one-hot expansion + a fail-closed coverage guard and asserts `state_dim == len(feature_names)` ([entry_iql_v2_adapter.py:136-148,200-252](gx1/runtime/entry_iql_v2_adapter.py#L136)) — so adding/removing a feature is a contract change the serve mirrors automatically (the clean λ-rebuild = same 197, byte-identical names+order to the active cement).
- **Zero-fill guard:** required feats (training std ≥ 1e-3) raise `RuntimeError` if absent; constants/harmless one-hots
  zero-fill silently — the 2026-05-19 LONG-bias guard ([entry_iql_v2_adapter.py:57-67,223-240](gx1/runtime/entry_iql_v2_adapter.py#L57)).
- **ctx dims contract-driven:** `ctx_cont_dim = CTX_CONT_DIM_V3` (105 regime-off / 123 REGIME_V4 — the live fase2b cement), `ctx_cat_dim = CTX_CAT_DIM_V3`
  (6 cement / 5 REGIME_V4, R4 2026-06-04 — was hardcoded 6); both locked at load ([v12_v10_live.py:115-118](gx1/execution/v12_v10_live.py#L115)).
- **Robust to ARBITRARY price level — verified 2026-06-21 (the "gold → 6000?" question).** Two layers: (1) the V10/V3/XGB 41-dim SEQ is 100% scale-invariant (returns-bps / `atr_z` / `atr_bps` / `*_atr` distances / `vol_z`/ratio/pct — NO raw level; [signal_bridge_v3.py:81-124](gx1/contracts/signal_bridge_v3.py#L81)); (2) the Entry-IQL z-scores ALL 197 inputs and **clamps to ±5σ, identical train+serve** (`z_clamp=5.0`, [entry_iql_multi_head_gpu_core_v1.py:152,160-161,242-244](gx1/scripts/entry_iql_multi_head_gpu_core_v1.py#L152)). The one heavily-weighted raw-price input `mid_canon_v1` (FOLD_1 mean 1833 ± 79) **saturates at +5.0 for any price ≥ ~2227** — so it has been pinned at +5.0 through all recent live trading and is bit-identical at 4285 / 6000 / 10000. No extrapolation/blow-up/NaN; nothing fail-closes on price level (only data-staleness does); drift stays advisory. Caveat: the model is therefore price-LEVEL-blind above ~2227 (flies on the scale-invariant feats) — a recement would re-center the normalization on a 4000+ world (and regen `drift_reference_v1.parquet`), but that buys calibration/selection finesse, NOT robustness (the clamp already guarantees that). [[project_gx1_direction_ceiling_openmore_20260616]]

## 11. Exit stack internals (V3 → Exit-IQL → overlays)

- **V3 has 4 heads:** should_exit_prob (main), profit_protect_prob, family_argmax (4-class), family_logit_max — Exit-IQL consumes all 4 ([v12_v3_live.py:8-11](gx1/execution/v12_v3_live.py#L8)). These are the **raw `v3_v8_*` block** (present in the cement's 209-feat state, fed live every M1 — [v12_pipeline.py:634](gx1/execution/v12_pipeline.py#L634)). NOT to be confused with the **7 `v3_*_v1` running-stats** (max-prob-since-entry, consecutive-exits, acceleration…) derived in [v12_trade_state.py:395-418](gx1/execution/v12_trade_state.py#L395) — a Phase-4/V12.1.1-vintage feature the CURRENT CLEAN cement was **not** built with.
- **`[EXIT_IQL_V3_TRACKING_MISSING]` warning is BENIGN (verified 2026-06-21):** the active `exit_iql_retrain_clean_20260609` bundle has all 4 raw `v3_v8_*` but none of the 7 running-stats → the load-time sentinel ([v12_exit_iql_live.py:355-363](gx1/execution/v12_exit_iql_live.py#L355)) fires by design. Serve still *computes* the 7 ([v12_pipeline.py:658](gx1/execution/v12_pipeline.py#L658)) but the featurizer drops any name absent from the bundle's `feature_names` → **train==serve preserved, no skew**. Companion `[EXIT_IQL_V3_BLOCK_PARTIAL]` (line 348) guards the raw block; does NOT fire (all 4 present). Re-adding the 7 = a vedtak-gated IQL refit (transformer frozen) via [augment_exit_iql_dataset_with_v3_tracking.py](gx1/scripts/augment_exit_iql_dataset_with_v3_tracking.py), NOT a fix — exit is the strong link.
- **V3 multi-TF MANDATORY live:** `transformer_config.json` must have `multi_tf.enabled=true`; single-TF bundles fail
  hard (`RuntimeError`) — prevents the COSTFIX-era silent V7-loaded-as-V6 path ([v12_v3_live.py:156-163](gx1/execution/v12_v3_live.py#L156)).
- **Exit-IQL reward = R_NET_REAL:** `r_hold = 0.5·hold_K − 0.5·|MAE| − 2·spread`; `r_exit = exit_now − 2·spread`
  (α=0.5 MFE, β=0.5 MAE, γ=2.0 spread). `hold_K = max(remaining intrabar peak[t+1:t+1+K])` — **peak-seeking, not close** ([materialize_build_exit_iql_v2.py:113-115](gx1/scripts/materialize_build_exit_iql_v2.py#L113), [..._per_bar_v1.py:343](gx1/scripts/materialize_build_exit_iql_per_bar_dataset_v1.py#L343)).
- **Exit-IQL state = 130-dim (cement):** 8 per-bar + ~109 ctx (12 entry + ~86 canon + 6 deriv + 5 snap) + ~14 one-hot ≈131, O5 dead-drop → 130 ([materialize_build_exit_iql_v2.py:150-572](gx1/scripts/materialize_build_exit_iql_v2.py#L150)).
- **Strategy-F (MFE-giveback overlay), `GX1_STRATEGY_F_ENABLED` default ON** — 4 sequential rules ([v12_exit_iql_live.py:72-551](gx1/execution/v12_exit_iql_live.py#L72)):
  1. profit-lock: MFE≥30bps & drawdown≥30%·MFE → EXIT
  2. break-even-cut: MFE≥10bps & pnl<30%·MFE → EXIT
  3. strong-hold: suppress 1+2 if IQL `Q_adv < −200`
  4. hold-horizon-expired: bars > 1.5×hold_pred & mfe<30bps (floor 60 bars) → EXIT
  All thresholds `GX1_*` ablatable (§17). Validated +136 bps WITHOUT Strategy-F but live runs it ON → OOT-ablate, default OFF post-retrain. ⚠ The +136 figure is corrupt-April-2026-INFLATED (~80%); honest post-x10-repair value ≈ **28 bps/take** (2026-06-06 correction) — re-measure on ALL repaired data before relying on it.
- **Distilled Q-swap (`GX1_USE_DISTILLED_EXIT=1`):** if `v3_v8_out` carries `v3_q_hold/exit`, swap IQL rec for distilled argmax; `decision_source='DISTILLED_V3_QHEAD'`; Strategy-F overlays on the swapped baseline ([v12_exit_iql_live.py:177-214](gx1/execution/v12_exit_iql_live.py#L177)).
- **Loaders fail-closed:** Exit-IQL bundle/variant/fold/aggregator from `gx1_guards.artifacts.load_decision_entry('exit_iql')` unless all 4 kwargs explicit ([v12_exit_iql_live.py:243-255](gx1/execution/v12_exit_iql_live.py#L243)); V10/V3 via `gx1_guards.load_decision_artifact()` (no hardcoded fallback) ([v12_v10_live.py:68-77](gx1/execution/v12_v10_live.py#L68)).

## 12. EXIT_IO contract lineage (V6 → V7 → V8) — prefix-stable

| io_version | feats | adds | file |
|---|---|---|---|
| V6 | 91 | base canonical M1L512 | exit_io_v6_ctx_v3canonical_m1l512.py |
| V7 | 155 | +4 volume +24 group-A +36 dip/struct | exit_io_v7_volume_dipstruct_m1l512.py:3-30 |
| V8 | **173** | V7(155) + **18** REGIME_V4 (regime class + trend-age/TF + cross-TF + change-detect) | exit_io_v8_regime_m1l512.py:1-30 |

- **Prefix-stable:** V7[:91]==V6, V8[:155]==V7 (runtime assertions) → warm-start from prior weights + **stable trade-state
  overlay indices 7-25 across all three**. Registry **defaults V6** if `io_version=None` (cement target = V7→V8; fail-close that None) ([registry.py:56-58](gx1/exits/contracts/registry.py#L56)). `SUPPORTED_V3_CONTRACTS` maps io_version → (features, count) ([v12_v3_live.py:80-151](gx1/execution/v12_v3_live.py#L80)).

## 13. XGB bridge + base80

- **base80 bundle = 80 feats** incl cyclic time (hour_sin/cos, dow_sin/cos); **4 session heads** (ASIA/EU/OVERLAP/US)
  trained separately, per-bar routing via `session_id` `{0→ASIA,1→EU,2→OVERLAP,3→US}` ([xgb_input_features_base80_v1.json](gx1/xgb/contracts/xgb_input_features_base80_v1.json), [v12_xgb_live.py:51-53,163-197](gx1/execution/v12_xgb_live.py#L51)).
- **7-dim bridge order** = `[p_long, p_short, p_flat, p_hat, uncertainty_score, margin_top1_top2, entropy]`;
  `entropy = -Σ p·ln(p)` (eps 1e-12, natural log); `margin = sorted_desc[-1... i.e. top1] - top2` ([signal_bridge_v3.py:62-68](gx1/contracts/signal_bridge_v3.py#L62), [xgb_multihead_model_v1.py:88-94](gx1/xgb/multihead/xgb_multihead_model_v1.py#L88)).
- **SEQ matrix = (n_m5, 41)** = 7-dim bridge + **34-dim price-state** (was 30; +4 volume `vol_z_20/vol_ratio_5_20/vol_pct_96/signed_vol_z_20` 2026-05-26). `SEQ_SIGNAL_DIM_V3=41` used by ALL V10+V3 training ([signal_bridge_v3.py:69,124,131](gx1/contracts/signal_bridge_v3.py#L69)). Volume 4 derived from raw `volume` via `volume_features.compute_volume_features` (one-truth) ([signal_bridge_v3.py:115-123](gx1/contracts/signal_bridge_v3.py#L115)).
- **Live XGB fail-closed on NaN:** `predict()` sets `allow_nan_fill=False`, sanitizer `hard_fail_on_nan=True` → `ValueError(SANITIZER_NAN_FAIL)` not 0-fill ([v12_xgb_live.py:186-193](gx1/execution/v12_xgb_live.py#L186)). Contract cross-check raises `[XGB_CONTRACT_MISMATCH]` on feature-name mismatch (B5 2026-06-04 added `feature_names_ordered` lookup) ([v12_xgb_live.py:99-126](gx1/execution/v12_xgb_live.py#L99)).
- **decision_indices = `arange(seq_len-1, n)`** (default 95..n): V10 only emits decisions where ≥`seq_len-1` history exists; these map filtered candidates back to global cv2 rows ([...candidates_v3_v1.py:246,421](gx1/scripts/materialize_inference_batch_candidates_v3_v1.py#L246)).

## 14. Multi-TF mechanics

- **`MULTI_TF_PER_BAR_FEATURES_V2` = 25 cols/TF** (9 V1 + 6 EMA-stack + 2 regime + 4 VWAP + 4 Boll-ADX). ONE truth
  `build_multi_tf_per_bar_features_v2`; V10 calls it direct, V3 build + serve via `attach_v2_mtf_per_bar_scalars` (2026-06-04 one-truth) ([htf_features.py:446-803](gx1/features/htf_features.py#L446)).
- **`get_last_n_at_or_before`** returns `(n, n_feat)` float32 **ZERO-LEFT-PADDED on warmup** (cold-start serve zero-pad =
  C/D skew vs long-history train tape — verify live tape carries ≥220 D1 / ≥80 H4). Cutoff = `target_ns − MULTI_TF_SHIFT`, searchsorted side=right; `.attrs` fast-path ~100× ([htf_features.py:860-910](gx1/features/htf_features.py#L860)).
- **FFILL before shift:** `compute_per_bar_features_v2` ffills EMA (20/50/100/200) + momentum close before shift — else
  sparse D1/H4 warmup → NaN→fillna(0)→dead constant feature ([htf_features.py:666-705](gx1/features/htf_features.py#L666)).
- **Multi-TF cache** loaded once from `--v2-cache-dir` (REQUIRED, no default; must match the cache the V2 V10 was trained on, NOT V1) ([...candidates_v3_v1.py:542-558](gx1/scripts/materialize_inference_batch_candidates_v3_v1.py#L542)).
- V3 builder recomputes the 12 `{tf}_*_v2` REGIME_V4 source cols FRESH, overwriting stale join values, to match serve ([materialize_build_v3_training_dataset_v2.py:336-358](gx1/scripts/materialize_build_v3_training_dataset_v2.py#L336)).

## 15. Build / dataset mechanics

- **Exit-IQL dataset skips t=0** (entry bar); rows from t=1 per `emit_stride`. v1 stride=4 (M5-equiv ~24-48 rows/trade),
  v2_m1 stride=1 (full M1). V3 builder `emit_stride=5`, `max_bars_per_trade=240` (4h) ([..._per_bar_v1.py:325](gx1/scripts/materialize_build_exit_iql_per_bar_dataset_v1.py#L325), [build_v3_dataset_v2:129](gx1/scripts/materialize_build_v3_training_dataset_v2.py#L129)).
- **Chronological split:** `GX1_EXIT_IQL_SPLIT_MODE` (chronological + 24h embargo, trade-grouped; default = cement bit-parity) — Entry-IQL default chronological+192-embargo since 2026-06-03; replaces the leaky `StratifiedShuffleSplit` (R16, 9d9c594f).
- **`R_WAIT_OPP_K96_LAM50`:** base/cement broadcasts single K=96 reward to all 6 Q-heads; `_SYM` variant (vedtak 2026-06-03)
  = per-K reward + 0 spread-coef (take_now terminal pnl already embeds spread) + symmetrized MAE on the waited side ([materialize_build_entry_iql_v2.py:537-582](gx1/scripts/materialize_build_entry_iql_v2.py#L537)).
- **Teacher labels** (V3): `teacher_final_pnl/mfe/mae/duration` written at FULL-trade horizon ([build_v3_dataset_v2:474-477](gx1/scripts/materialize_build_v3_training_dataset_v2.py#L474)).

## 16. Known live-vs-backtest gaps / gotchas (don't re-discover these)

- **Entry-bar cadence + entry-price basis (R13 RUN-gate).** Builder overlay row 0 = the **entry bar** (`entry=ask_open[s_t]`,
  bars_held 0-based). Live opens `TradeState.open(entry_bid=bid, entry_ask=ask)` (decision-bar **close**) and first `update_bar` runs the **next** minute → ~1-bar offset + different entry-price basis. PRE-EXISTING (old MVP too). Verified by R13 "per-feature parity assert on a replayed trade" (RUN-gated, post-rebuild).
- **Entry safety is ALWAYS-ON** (not PURE_PHASE6-gated): `evaluate_entry_safety` → `circuit_breaker_v1.evaluate_same_opp_cap`
  enforces no-short-in-long + no-same-side pile-up, reconciling local TradeState vs broker `get_open_trades()` — defends the **2026-06-02 −2000 USD incident** (16 shorts stacked after PURE_PHASE6=1 disabled the cap) ([v12_paper_runner.py:157-174,711-744](gx1/execution/v12_paper_runner.py#L157)).
- **Loop cadence:** runner polls ~10s but decides **once per M1** (`last_decision_minute==current_minute` gate); `update_bar`
  increments `bars_in_trade` once/minute ([v12_paper_runner.py:507-515](gx1/execution/v12_paper_runner.py#L507)). **24h hard cap:** `bars_in_trade>=1440` → `FORCED_CLOSE_24H` ([:655-660](gx1/execution/v12_paper_runner.py#L655)).
- **Counterfactual daemon two tracks:** Track A (forward-outcome) hourly, only journals **>25h old** (K=1440 forward + 1h margin) — since 2026-06-12 it also judges every FILLED take (`judge_take`: false_take / wrong_side / held_too_short / mfe_giveback + correct_skip → per-day `trade_verdicts_*.jsonl` regret feed); Track B (variant-shadow) every 10min on fresh journals, **`--variants auto`** = enumerate the contract-resolved bundle's own checkpoints (a bundle flip can no longer silently empty the shadow; empty per_variant = exit 1 fail-loud) ([v12_daily_counterfactual.sh:36-66](gx1/execution/v12_daily_counterfactual.sh#L36)).
- **IN-PROCESS SHADOW (ladder 2026-06-12):** `GX1_SHADOW_BUNDLE_DIR` (armed by the launcher from `GX1_DATA/config/shadow_bundle_dir.txt`) loads a SECOND Entry-IQL adapter in the runner that scores every poll through the live `predict()` path (incl. conviction-gate env → candidate-vs-active on the live operating point) and journals `shadow_action`/`shadow_q_per_action`/`shadow_agrees_with_live` (also on cluster-1-blocked polls). Fail-SAFE by design: variant/fold auto-resolve from the candidate's own checkpoints, 3-strike disable journaled as `shadow_disabled_reason` (never confusable with "not armed"), never affects the live decision ([v12_pipeline.py `_load_shadow_entry_iql`/`_attach_shadow_fields`](gx1/execution/v12_pipeline.py)). Rotated to the newest gate-PASSING nightly candidate; picked up at runner restart.
- **NIGHTLY LEARNING LOOP:** `gx1-nightly-learning.timer` 03:30Z → [scripts/gx1_nightly_learning.sh](scripts/gx1_nightly_learning.sh): verdict-accumulation → canonical-tape freshener (M1 OANDA-history backfill + M5 downsample; idempotent, fail-loud chunk guard) → ENTRY+EXIT replay buffers (matured D-8..D-2; entry = cement-M5 label convention, exit = 209-dim transitions dataset-only) → KS drift (advisory, name-aligned vs `drift_reference_v1.parquet` in the ACTIVE bundle) → [standing-vedtak-gated, rule 3] 3-fold warm-start refit w/ `--mix-cement` anchor (`cement_replay_sample_v1.parquet`) → D-1 out-of-sample shadow report → [scripts/gx1_candidate_gate.sh](scripts/gx1_candidate_gate.sh) `--quick` → shadow rotation ONLY on gate PASS. Gate verdicts carry a `wave_caveat` (entry decisions inferred on the EXIT wave's forward_outcome — same pairing as cement evidence; entry-wave-keyed inputs = tracked follow-up per the 2026-06-11 wave-mismatch lesson); FULL mode (decisive volbal-baseline posthoc) required before any contract flip. Promote = manual `PROJECT_STATE_artifacts.json` flip, always.
- **systemd vs launcher interval mismatch:** `gx1-canonical-incremental.service` runs `--interval 15`; `launch_live_practice.sh:146-147` runs `--interval 60`. **systemd is source of truth.**
- **Stale multi-TF cache footgun (build-pipeline only as of 06-08):** the V10/V3 BUILD scripts default `GX1_V10_MULTI_TF_V2_CACHE_DIR` to a hardcoded path → a stale cache builds fresh-cv3 datasets wrong. `build_context()` raises `[MTF_CACHE_STALE]` if the cache lags the M5 cutoff by > 2 days (see §19); a Fase-2B-style rebuild MUST regen `prebuild_multi_tf_cache_v2` + set the env var. **LIVE serve no longer reads this cache** (§6, in-memory) — so the stale guard can no longer break live; it only guards builds.
- **Entry Transformer full feature audit (2026-06-26):** run `python -m gx1.audit.entry_transformer_feature_audit` before V10 retrain. It covers all 294 named input surfaces (41 seq/snap + 123 ctx_cont + 5 ctx_cat + 5×25 MTF) and writes CSV/JSON under `/home/andre2/GX1_DATA/reports/entry_transformer_feature_audit_20260626*`. Current findings: old 6yr train parquet has corrupt `body_pct` snap/seq_last max `1.196e10`; bodyfix rebuild at `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_bodyfix` passes `body_pct_max=1.0` and audit bounds. Remaining: `smc_choch` is a too-sparse flip-bar bit (~0.1% nonzero), D1 regime/structure cols can be constant in OOT windows, `y_hold_horizon_target` is neutral 0.5 because the historical Exit-IQL per_bar root is missing. Bundle metadata ctx-name length mismatch (21 vs 123) was fixed in writer and existing active/candidate metadata (backups `.bak_ctx21_20260626`). Spread dead-feature root cause was fixed after the bodyfix A/B: `add_ctx_cont_columns_to_prebuilt` now derives `spread_bps` from `bid_close/ask_close` like live and clamps negative bid/ask glitches to 0; `scripts/v10_6yr_rebuild_20260626.sh` now defaults to `v10_6yr_rebuild_20260626_spreadfix` and fail-closes if `FULL_PLUS_CTX` still has constant spread despite bid/ask signal. Spreadfix dataset materialized at `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix`: `FULL_PLUS_CTX` spread gate mean `1.7366`, p95 `2.6043`, std `0.7436`; audit risk rows `41` (was `50` on bodyfix), and `spread_bucket` now has 5 categories on train/val/test. The audit also writes `derived_feature_candidates.csv` with no contract change; current top candidate families are SMC recency/pressure (`smc_choch_recent_tau24`, sweep/BOS pressure), support/resistance proximity (`sr_nearest_pivot_abs_atr`), and dip/structure aggregates. Old bodyfix A/B did not beat cement and still used stale zero-spread: baseline bundle `v10_bundle_6yr_baseline_clean` best val ACC `0.4344459`, test ACC `0.4750165`; symmetric-negatives bundle `v10_bundle_6yr_symneg` best val ACC `0.4544726`, test ACC `0.4789786`. Both strict-load and post-export liveness passed; do not promote without a spreadfix retrain plus loss/head/horizon gate evidence.
- **Spreadfix V10 retrain result (2026-06-26):** Entry Transformer was the correct first retrain target (XGB remains diagnostic/feature-ranking). Spreadfix+symneg bundle `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_bundle_6yr_symneg` trained on clean git `2a078092`, strict-loaded, and post-export feature-liveness passed (`all inputs alive/allowlisted`). Metadata: best epoch `4`, `best_dir_acc=0.4592790`, `best_val_loss=5.9865565`, last epoch `10`; `EVAL_TEST.json`: `test_acc=0.4785384`, `test_loss=1.8266977`, `n=4543`. This beats bodyfix symneg on validation but not on test (`0.4785384` vs `0.4789786`) and remains below the cement-level target (~`0.524`), so **do not promote**. Run evidence points to training-recipe issues before more blind retrains: val overfit after epoch 4, repeated `clean_edge~survival/tradable` aux-head collapse warnings, and session skew in test long handling (ASIA long pred rate `0.191`, OVERLAP `0.674`, US short-on-long rate high).
- **V10 trainer default guard fix (2026-06-26):** `entry_v10_ctx_train_v3.py` had stale `_env_str` defaults that silently diverged from `_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS`; most importantly `ENTRY_AUX_BAD_PATH_WEIGHT` defaulted to `0.30` although the canonical contract said parked `0.0`, so bad-path trained active unless env overrode it and produced repeated anti-target warnings. Fixed defaults now match the canonical guard (`direction_ce=1.30`, tradable `1.15`, bad_path `0.0`, clean_edge `0.45`, survival `0.10`, rank `0.25/0.12`), with `tests/test_entry_v10_train_defaults.py` preventing drift. Existing bundles are unchanged; next V10 retrain should rerun spreadfix+symneg on this corrected recipe before any XGB-side retrain.
- **Corrected-defaults V10 retrain result (2026-06-26):** rerun spreadfix+symneg to `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_bundle_6yr_symneg_guardfix` after the default-guard fix. Recipe confirmed `bad_path_weight=0`, strict-load and feature-liveness passed, but metrics did not improve: best epoch `4`, `best_dir_acc=0.4536716`, `best_val_loss=5.3548960`; test `acc=0.4783183`, `loss=1.6734635`, `n=4543`. Lower loss but flat/worse ACC vs previous spreadfix (`0.4785384`) and bodyfix symneg (`0.4789786`) means **do not promote** and stop same-recipe V10 retrains. Next ACC work should be diagnostic: XGB/linear ablations for feature families, primary horizon/label audit, and/or a V10 contract feature wave from `derived_feature_candidates.csv` (SMC recency/pressure, SR proximity, dip/structure aggregates) with train+serve parity.
- **Entry Transformer feature-family diagnostics (2026-06-26):** `python -m gx1.audit.entry_transformer_feature_diagnostics` builds a non-production tabular view from the same dataset surfaces (snap + seq summaries + ctx_cont/cat + MTF current + audit-only derived candidates) and writes `/home/andre2/GX1_DATA/reports/entry_transformer_feature_diagnostics_20260626_spreadfix`. XGB diagnostic (`80k` train sample, `482` features) reached only val/test ACC `0.43765/0.46489`, below transformer, so XGB is diagnostic only. Strongest standalone families on test: `ctx:volatility_range` `0.44816`, `ctx:momentum_trend` `0.43121`, `mtf_current:volatility_range` `0.41867`; feature-importance is dominated by vol/regime/session/XGB-bridge (`vol_regime_id`, `_v1_atr14`, `rvol_20`, `seq_mean12 p_short`, `p_long`, `atr_bucket`, `atr_bps`). Derived candidates are not standalone lift yet (family-only = majority `0.3819`); nonzero derived importances were mainly `sr_support_minus_resistance_prox`, `struct_tf_agree_count_v3`, `dip_proximity_mean_h1h4d1`, `smc_bos_pressure_last48`. Horizon diagnostic confirms primary label is **3 M5 bars** and path-quality horizon **10 bars**; forward-return sign alignment improves materially with longer K on OOT (`K1=0.368`, `K5=0.441`, `K12=0.512`, `K24=0.613` test), so next ACC work should test a longer/swing-aware direction target or aux target, not keep optimizing the same 15-minute y_direction.
- **LAG TRUTH — exit PRICE reaction vs FEATURE freshness (don't conflate; 2026-06-08).** The price the bot trades on is **live-polled every ~10s** straight from OANDA `client.get_pricing()` with a `StaleQuoteError` guard rejecting quotes older than `DEFAULT_QUOTE_MAX_AGE_SEC` ([v12_paper_runner.py:116-136,468](gx1/execution/v12_paper_runner.py#L116)) → already ≤~10s, satisfies "≤1 min". The **collector parquet** (`m1_close` + canonical features, now ~15s) and the **~280s async-refresh** (model-input feature context: cv3 M5 feats, multi-TF, base34) are a SEPARATE, slower path — that's where "1–2 min lag" lives, and it gates FEATURES not the trade price. Sub-1-min *price reaction* is already met by polling; tick streaming would mainly sharpen `m1_close`/microstructure, not the live quote.
- **Logger timestamps are +2h vs real UTC (cosmetic TZ bug):** live logs stamp `…THH:MM:SSZ` in local CEST, not UTC. A log line "09:25Z" = real 07:25 UTC. Compare DATA timestamps (true UTC) to `date -u`, NOT to log stamps, when judging lag — else you'll misread a +2h offset as a 2-hour data lag.
- **float32 vs float64** (overlay): ≤~5e-6 bps on 2nd-diff cols — harmless. **Two ATRs** (§5). **REGIME_V4:** build/serve defaults differ (build 1 / serve 0); the launcher PINS `GX1_REGIME_V4=1 GX1_TREND_REGIME_FROM_D1=1 GX1_EXIT_AUGMENT_64=1` for the **fase2b cement** (commit c1cca55d) — was pinned 0 for the prior COSTFIX cement. Flip in lockstep with the active contract. **trend_regime_id DROPPED** (R4) — don't reintroduce. **Exit M1 sacred** — never coarsen to M5 (hook blocks).

## 17. Flags index (set explicitly in run-manifests; never trust defaults)

| flag | meaning | default |
|---|---|---|
| `GX1_REGIME_V4` | regime-v4 ctx (cont 123/cat 5 vs 105/6) | build/contract `1`, serve/cand `0`, launcher PIN `1` (fase2b cement) |
| `GX1_CONVICTION_GATE` / `GX1_CONVICTION_THR` | entry conviction-gate: open by `raw_adv` (≥ thr) overriding IQL SKIP | code OFF / `-34.2`; launcher PIN `1` / **`-37.71`** (open-more-to-−100 wave REVERTED; was -34.2/top-20% @06-10) |
| `GX1_ENTRY_MARGIN_FLOOR` / `_THR` | entry margin-floor overlay: SKIP a final TAKE whose V10 margin < THR (never SKIP→TAKE) | code OFF / `0.47`; launcher UNSET (default-OFF; commit 0d9e74b0) |
| `GX1_SIZING_MODE` / `GX1_SIZING_CONV_SRC` / `GX1_SIZING_MARGIN_POW` / `GX1_SIZING_ATR_REF_BPS` | position sizing `clip((margin^POW/REF)×min(ATR_REF/atr,1),0.5,2)` (REF MUST track POW — footgun) | code `off`/`raw_adv`/`2.0`/`18`; launcher PIN `both`/`margin`/`2.0`/`14` (REF=0.3318) |
| `GX1_SKIP_ASIA` | block ASIA-session entries (per-year win floor) | code OFF; launcher PIN `1` |
| `GX1_PURE_PHASE6` | disable live-only wrappers (live=Phase6 1:1); CLUSTER1 stays ON; safety always-on | `1` for paper-runner |
| `GX1_STRATEGY_F_ENABLED` | MFE-giveback exit overlay (4 rules) | `True` (OOT-ablate → OFF post-retrain) |
| `GX1_MFE_GIVEBACK_PCT` / `_MIN_MFE_BPS` | profit-lock thresholds | `0.30` / `30.0` |
| `GX1_BREAKEVEN_*` / `_STRONG_HOLD_QADV` / `_HOLD_HORIZON_*` | other Strategy-F thresholds | ablatable |
| `GX1_USE_DISTILLED_EXIT` | swap Exit-IQL rec for V3 distilled Q | off |
| `GX1_EXIT_AUGMENT_64` | emit AUG64 bare-name canon feats (V8-train) | off |
| `ADAPTIVE_MIN_ADV_ATR_MULT` / `_FLOOR_BPS` | runtime entry adv gate | `0.35` / `1.5` |
| `GX1_EXIT_IQL_SPLIT_MODE` | chronological embargo split | chronological (cement-parity) |
| `GX1_MAX_PREBUILT_STALENESS_MIN` | fail-closed SKIP if prebuilt older | `30` |
| `GX1_V10_BUNDLE_DIR` / `GX1_V3_BUNDLE_DIR` | bundle override (else guard-driven) | guard |
| `GX1_V10_MULTI_TF_V2_CACHE_DIR` | multi-TF V2 cache dir for V10/V3 builds | env (regen + set at rebuild) |
| `GX1_MTF_CACHE_MAX_LAG_DAYS` / `GX1_MTF_CACHE_ALLOW_STALE` | stale-cache guard (§19) | `2` / off |
| `--vedtak <id>` | REQUIRED for any retrain (gx1_guards fail-closed) | — |

## 18. Protected core — warn-only (marker gate REMOVED 2026-06-05)

Protected dirs (CLAUDE.md rule 1): `gx1/execution`, `gx1/contracts`, `gx1/exits/contracts`, `gx1/models/entry_v10`,
`gx1/core`. The one-shot `ALLOW_CORE_EDIT` marker gate was **REMOVED 2026-06-05** (commit 70d22bbf, user vedtak — the per-edit `touch` friction was killing the workflow). The PreToolUse hook ([.claude/hooks/guard_write.py:86-97](.claude/hooks/guard_write.py)) now **WARNS** on protected-core edits — emits `additionalContext` ("⚠ PROTECTED-CORE EDIT…"), exit 0, edit allowed but loudly logged; the discipline (verify in-use, ONE truth, minimal change, train==serve) is unchanged. The ONLY remaining HARD BLOCK (exit 2) is M1→M5 coarsening of the exit's M1 grid in exit files ([guard_write.py:65-79](.claude/hooks/guard_write.py)). Bundle loads fail-closed via `gx1_guards`.

## 19. Retrain entrypoints + --vedtak gates (rule 3)

**Rebuild ordering/inputs/guards/dims** → [FASE2B_REBUILD_ORDER.md](FASE2B_REBUILD_ORDER.md) (the exact Fase-2B
sequence + the x10-fix + the add_ctx_cont guard sequence; orchestrated by `scripts/fase2b_rebuild.sh`). Read it
before running/resuming the rebuild — don't re-scan the chain.

Every model trainer calls `gx1_guards.gates.require_retrain_vedtak(args.vedtak)` right after `parse_args` (fail-closed; missing `--vedtak` aborts). Active trainer + dataset builder per model:

| model | active trainer | gated |
|---|---|---|
| XGB v7 base80 | train_xgb_universal_multihead_v2.py (req `--canonical-prebuilt-parquet`) | ✅ 06-05 |
| V10 entry | entry_v10_ctx_train_v3.py · build_entry_v10_ctx_training_dataset_v3.py | ✅ |
| Entry-IQL | materialize_build_entry_iql_v2.py | ✅ |
| V3 exit transformer | train_exit_v6_disk_thin.py (req `--dataset-dir` 06-05) · train_exit_transformer_v0_sharded.py | ✅ 06-05 |
| Exit-IQL | materialize_build_exit_iql_v3_m1.py (active) / _v2.py (legacy) · per_bar_dataset_v2_m1.py | ✅ 06-05 |

- **Multi-TF cache freshness:** `build_context()` raises `[MTF_CACHE_STALE]` if the V2 cache lags the M5 build cutoff by > `GX1_MTF_CACHE_MAX_LAG_DAYS` (default 2) — covers all 3 build paths ([augment_forward_outcome_v2.py](gx1/scripts/augment_forward_outcome_v2.py)). Rebuild must regen `prebuild_multi_tf_cache_v2` + set `GX1_V10_MULTI_TF_V2_CACHE_DIR`.
- `add_ctx_cont` manifest records `regime_v4_emitted` (the 18 REGIME_V4 cols are emitted by NAME, not counted in `ctx_cont_dim` {2..16}).
- **Pre-retrain blockers (2026-06-05 audit) — status 2026-06-11:** x10 2026-04 **RESOLVED for the build/train chain** — April-2026 x10 was rebuilt from the clean tape and the CLEAN x10-April-repaired chain cemented + flipped 2026-06-10 (`PROJECT_STATE_artifacts.json` note); the detect-guard (`price_glitch_guard`) remains in place · cv3 pin + stale BASE28 seed **RESOLVED by the fase2b rebuild** (fresh cv3 re-pinned `_PINNED_FASE2B_20260605`, fresh BASE28/base34/base80 built — see FASE2B_REBUILD_ORDER.md; data daemons running again under systemd, verified active 2026-06-11) · R13 parity-RUN (status as of 2026-06-05, re-verify). See FASE2_PREFLIGHT_RUNBOK.
- **R12 = NO protected edit needed** (traced 06-05): serve M1-native branch `v12_v3_live:307-318` is already correct + self-activates. The open piece is EDITABLE rebuild-prep: the full base34 (M1-expanded) rebuild must emit raw M1 `volume` on ALL rows (the incremental daemon does it for NEW rows only; no full-rebuild builder emitting volume was found — `materialize_build_extended_base34` builds M5 w/o volume, `canonical_prebuilt_rebuilder` has none). Either a one-shot volume backfill onto base34 or wire the rebuild to emit it.

---

## Maintenance rule (this is why we stop re-deriving)

**Before** tracing the chain or answering a train/serve / data-flow / parity question, READ THIS FILE.
**After** you derive any non-obvious fact the map didn't have (a new call site, a formula, a flag, a
gotcha, a moved file), ADD it here in the SAME session, with a `file:line`. Keep it TIGHT — facts +
pointers, not prose; this is a map, not a log (logs live in DECISION_LOG.md / PROJECT_STATE.md).
When code moves, fix the pointer here in the same change. One truth, fail-closed, current.
