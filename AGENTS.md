# GX1 Agent Guardrails

## What GX1 is — read before acting
- **FIRST, read [SYSTEM_MAP.md](SYSTEM_MAP.md)** — the ONE-TRUTH map of how the live chain wires
  together: data flow, train↔serve parity formulas, file→responsibility index, flags, and the
  live-vs-backtest gotchas. It exists specifically so we STOP burning a whole session re-deriving the
  same overview. If a "how is this wired / where does this come from / does build match serve" answer is
  NOT there, derive it once and ADD it to SYSTEM_MAP.md the SAME session (its Maintenance rule). Trust
  the map's `file:line` pointers; fix them in the same change when code moves.
- Solo-built XAUUSD trading bot: cooperating number-AIs. XGB produces per-M5
  bridge probs (p_long/p_short/p_flat) — every 5 min, NOT per minute — and feeds
  BOTH stages:
    - ENTRY (per M5 bar): XGB bridge -> V10 entry transformer -> Entry-IQL
      (final 3-action: SKIP / TAKE_LONG / TAKE_SHORT — no WAIT). Context = multi-TF M5/M15/H1/H4/D1.
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
- **Verify on the LOAD-BEARING metric, not blanket accuracy.** The bot trades a SELECTED high-conviction subset, so
  grade the metric it actually acts on (high-conviction-tail dir-acc, selected-trade win-rate / bps-take), not the
  blanket average. Blanket OOT can HOLD while the load-bearing tail regresses — 2026-06-05 the fase2b XGB held blanket
  0.508≥cement but its top-5% conviction collapsed 0.90→0.63, missed by a blanket-accuracy "verified" check.
- **NOTHING ignored — feature-liveness is ALWAYS checked (user vedtak 2026-06-06).** Every input feature the chain
  consumes must be ALIVE (non-constant on a SHUFFLED sample — a consecutive batch false-flags slow-varying feats like
  D1 regime) or on the documented allowlist. `gx1.audit.feature_liveness` is the ONE-TRUTH cross-chain check (XGB gain +
  V10 ctx/snap variance + multi-TF integrity: all 5 TFs present, alive, DISTINCT resolutions, ATR scaling sane). It runs
  AUTO at every V10 retrain (trainer post-export, fail-loud) and MUST pass before any cement
  (`python -m gx1.audit.feature_liveness --strict --v10-bundle … --xgb-bundle …`). Adding a feature to
  `KNOWN_ALLOWED_DEAD` requires a documented reason (structural/benign). A NEW dead feature = a silent-ignore
  regression → fix or document, never ship silently.

## ONE gjeldende — artifact selection (no version roulette) [CLAUDE.md rule 8]
- **ONE truth = `PROJECT_STATE_artifacts.json`** (repo root). It names the single ACTIVE artifact per role
  (xgb / v10_entry / v3_exit / entry_iql / exit_iql, + active_variant/folds/aggregator). Edit it ONLY via an
  explicit vedtak. It SUPERSEDES `GX1_DATA/CURRENT_BUNDLES.md` (already renamed `.SUPERSEDED_SEE_PROJECT_STATE`).
- **Resolve only through the contract.** Every build/decision/serve path loads its bundle via
  `gx1_guards.load_decision_artifact` keyed on the contract — NEVER by glob, `sorted(...)[-1]`, mtime/"latest",
  or a hardcoded default path. Missing/ambiguous/PENDING ⇒ raise, never silently fall back to an on-disk vintage.
  (Known footguns flagged 2026-06-03: a V3 resolver that fell back to a stale V9 on disk, and a build-path V9
  substitution — these are the exact pattern to hunt and kill; verify none remain before trusting "can't run wrong".)
- **New artifact lifecycle:** build → PENDING_VEDTAK → pass gates → I flip the contract to ACTIVE → the prior
  ACTIVE moves to `history[]` (INVALIDATED). Never two ACTIVE per role. Never auto-promote.
- **Physical de-duplication (so we don't drown in v1/v2/v3…):** superseded artifacts are removed via rule 5
  (backup → inventory → dry-run → user confirm → delete), QUARANTINE-first (reversible) e.g. `runs/_SUPERSEDED_<date>/`.
  NEVER delete the live-ACTIVE bundle while it is active, and NEVER mid-rebuild delete the chain a pending retrain
  still depends on. Cleanup of the OLD cement happens only AFTER the new chain cements + the contract flips.
- **The guarantee** is the fail-closed resolver refusing to guess — NOT physical deletion. Deletion reduces
  clutter; the resolver is what makes running-wrong impossible. Both are required; the resolver is load-bearing.

## Models & architecture
- V10/V3 transformer input contracts are SACRED. Never refactor a contract to fit upstream XGB changes — retrain XGB on the contract V10/V3 expects, or shelf the experiment.
- Multi-TF is ALWAYS mandatory — for V10 (entry) AND V3 (exit). Never single-TF.
- Exit is ALWAYS M1 resolution. Never coarsen/downsample (e.g. M1→M5-ffill) to save compute. Never reorganize existing architecture — extend at native resolution; gain speed via vectorization/numba/GPU.
- All smart AI, no hardcoded decision rules (e.g. no hardcoded relabel rules — let the model learn).

## Workflow discipline
- ONE-TRUTH OVERVIEW: consult [SYSTEM_MAP.md](SYSTEM_MAP.md) before tracing the chain or answering any
  data-flow / train↔serve / parity question. When you derive a non-obvious fact it lacks (a call site, a
  formula, a flag default, a gotcha, a moved file), write it back to SYSTEM_MAP.md the same session — the
  goal is that the NEXT session never re-derives it. Map = tight facts+pointers; logs stay in DECISION_LOG/PROJECT_STATE.
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
  2. `v12_canonical_incremental --loop --interval 15` — appends new M1 → canonical_v3 + BASE34 prebuilts; without this, cv3 cutoff falls behind and the paper runner clips `effective_ts` to a stale bucket → live becomes a frozen replay.
  3. `v12_paper_runner` with `GX1_PURE_PHASE6=1` — disables every live-only wrapper (TIME_OF_DAY_EXIT, ADAPTIVE_MIN_ADV, REGIME, PORTFOLIO_*, LOW_CONFIDENCE, spread cap) so live = Phase 6 OOT 1:1. NOTE: CLUSTER1_RATE_LIMIT is NO LONGER disabled by PURE_PHASE6 — since 2026-06-02 it is ALWAYS ON as a live sanity-floor (v12_pipeline.py:318-324); override only via GX1_CLUSTER1_DISABLE=1 for explicit OOT-replay runs.
  4. `v12_daily_counterfactual.sh --daemon` — every hour looks for journals older than 25h that haven't been replayed yet; runs `v12_counterfactual_replay.py` on each + writes per-day "skulle/skulle ikke handlet" report to `GX1_DATA/reports/v12_paper_runs/counterfactual_reports/`. Idempotent via marker files in `.replayed_markers/`.
- Stop cleanly with `bash scripts/stop_live_practice.sh` before code edits that touch live runtime.
- The data daemons (collector + canonical_incremental) run under **systemd --user** (`gx1-collector.service`, `gx1-canonical-incremental.service`) and log to `/home/andre2/GX1_DATA/reports/v12_paper_runs/logs/{collector,canonical_incremental}.log`. The `launch_live_practice.sh` nohup fallback (which would log to `/tmp/gx1_live_practice/`) is NOT the live source — prefer the systemd units (`systemctl --user status gx1-collector gx1-canonical-incremental`).

## Git & secrets
- Never amend live commits. Never force push.
- Never commit secrets (`.env`, credentials).
