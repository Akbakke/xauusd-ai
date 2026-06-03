# GX1 Agent Guardrails

## What GX1 is — read before acting
- Solo-built XAUUSD trading bot: cooperating number-AIs. XGB produces per-M5
  bridge probs (p_long/p_short/p_flat) — every 5 min, NOT per minute — and feeds
  BOTH stages:
    - ENTRY (per M5 bar): XGB bridge -> V10 entry transformer -> Entry-IQL
      (final take-vs-wait). Context = multi-TF M5/M15/H1/H4/D1.
    - EXIT (per M1 bar, in-trade): the latest M5 XGB bridge asof-filled onto each
      M1 bar (held for 5 M1 bars; `m5_phase_0..4` = minutes since the XGB refresh)
      + V10 entry-snapshot + trade-state (mfe/mae/dd_from_mfe/giveback/...)
      -> V3 exit transformer -> Exit-IQL (+ MFE-giveback overlay).
      M1-NATIVE price/trade-state, 512-bar window + same MTF x5.
- XGB stays M5 (trained on M5 canonical features); it is NOT recomputed per M1.
  Exit gets M1 resolution on PRICE + trade-state; the XGB directional context
  refreshes at M5 with `m5_phase` encoding staleness. NEVER coarsen/resample the
  exit's M1 price grid to M5.
- Decisions are data-driven + OOT-validated, never in-sample.
- WORKING MODE: extend/modify the EXISTING component. Never rebuild from scratch
  or fork a parallel script. If a script/contract/helper exists, change THAT one.
  New files only for a genuinely new shared one-truth helper — never to duplicate
  existing logic. Before creating any file: name the existing file you considered
  extending and why it didn't fit.

## Environment
- Use `/home/andre2/src/GX1_ENGINE/.venv/bin/python` for Python commands.
- Treat `PROJECT_STATE.md` as the current local project state.

## Scope & isolation
- XAUUSD only. No other instruments in this project — never share data, models, paths, configs, reports, code trees, or memory with anything else.

## Decisioning data integrity
- Never use dummy, synthetic, or degraded fallback inputs for decisioning.
- Never use implicit latest/glob artifact selection for decisioning.
- Never use in-sample scores as decision-valid evidence.
- Never select old invalidated V3 artifacts for decisioning.

## Models & architecture
- V10/V3 transformer input contracts are SACRED. Never refactor a contract to fit upstream XGB changes — retrain XGB on the contract V10/V3 expects, or shelf the experiment.
- Multi-TF is ALWAYS mandatory — for V10 (entry) AND V3 (exit). Never single-TF.
- Exit is ALWAYS M1 resolution. Never coarsen/downsample (e.g. M1→M5-ffill) to save compute. Never reorganize existing architecture — extend at native resolution; gain speed via vectorization/numba/GPU.
- All smart AI, no hardcoded decision rules (e.g. no hardcoded relabel rules — let the model learn).

## Workflow discipline
- NEVER auto-retrain. Get an explicit user decision (vedtak) before every retrain.
- Check existing code before building; keep ONE truth (no duplicated/overriding logic); fail-closed defaults; minimal change; clean up superseded artifacts as you go; run a post-task bug/mismatch hunt.
- Always maximize CPU/GPU/RAM utilization (raw + smart prefetch/numba/GPU).
- Do not run R6, freeze, promo, live, or package build without an explicit green gate.
- Keep historical artifacts as history unless an explicit selection contract marks them active.

## Running live practice
- One launcher script for the whole stack: `bash scripts/launch_live_practice.sh`.
- Idempotent — reads `*.pid` files in `GX1_DATA/reports/v12_paper_runs/`, skips anything already alive, starts only what's missing. Re-run any time to verify the stack is up.
- Starts four components together (they must ALL be running for live to track Phase 6 cement + auto-report):
  1. `v12_oanda_data_collector` — pulls M1 OHLC from OANDA practice every 60s.
  2. `v12_canonical_incremental --loop --interval 60` — appends new M1 → canonical_v3 + BASE34 prebuilts; without this, cv3 cutoff falls behind and the paper runner clips `effective_ts` to a stale bucket → live becomes a frozen replay.
  3. `v12_paper_runner` with `GX1_PURE_PHASE6=1` — disables every live-only wrapper (TIME_OF_DAY_EXIT, ADAPTIVE_MIN_ADV, REGIME, PORTFOLIO_*, LOW_CONFIDENCE, spread cap, CLUSTER1_RATE_LIMIT in v12_pipeline) so live = Phase 6 OOT 1:1.
  4. `v12_daily_counterfactual.sh --daemon` — every hour looks for journals older than 25h that haven't been replayed yet; runs `v12_counterfactual_replay.py` on each + writes per-day "skulle/skulle ikke handlet" report to `GX1_DATA/reports/v12_paper_runs/counterfactual_reports/`. Idempotent via marker files in `.replayed_markers/`.
- Stop cleanly with `bash scripts/stop_live_practice.sh` before code edits that touch live runtime.
- Logs land in `/tmp/gx1_live_practice/{paper_runner,oanda_data_collector,canonical_incremental}.log`.

## Git & secrets
- Never amend live commits. Never force push.
- Never commit secrets (`.env`, credentials).
