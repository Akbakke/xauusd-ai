# ACTIVE ENTRY FOUNDATION OVERRIDE - 2026-06-28

Read `docs/ENTRY_FOUNDATION_AUDIT_20260628.md` and
`docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md` before acting on
Entry work. They supersede older XGB/V10/Entry-IQL live-practice instructions in
this file. The 2026-06-27 no-XGB shadow plan is historical pre-foundation
evidence, not the active operating point.

Use only the canonical control surface for this wave:
- `scripts/entry_next_edge_control.sh verify`
- `scripts/entry_next_edge_control.sh selftest`
- `scripts/entry_next_edge_control.sh foundation-guardrails`
- `scripts/entry_next_edge_control.sh worktree-hygiene`
- optional, explicit cleanup staging:
  `scripts/entry_next_edge_control.sh stage-foundation-cleanup --apply --vedtak <id>`
- `scripts/entry_next_edge_control.sh train-readiness`
- after explicit user vedtak only:
  `scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit`

Readiness is `READY_FOR_VEDTAK_SMOKE_TRAIN` only when foundation gates and
git-clean execution hygiene both pass. `READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN`
means the foundation contract is ready but real trainer start is still blocked.
Candidate training, replay-readiness, IQL distillation, IQL replay comparison,
promotion review, shadow, and live remain closed until their preceding
foundation gates produce explicit PASS/READY artifacts.

Do not start legacy live/practice/OANDA order paths, old V10/XGB/ET research
launchers, direct `v12_paper_runner`, or promotion/pin/live actions without the
foundation-specific gate and explicit user vedtak.

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
- Current state = `PROJECT_STATE_artifacts.json` (the ONE selection truth) + `bash scripts/gx1_handover.sh`
  (live overview). `PROJECT_STATE.md` is a frozen 2026-04-30 historical log (superseded banner inside).

## Scope & isolation
- XAUUSD is the only TRADED instrument. No other instruments are traded, and never share data, models, paths, configs,
  reports, code trees, or memory with any OTHER PROJECT.
- **CROSS-ASSET DATA AS FEATURES is permitted (user vedtak `cross_asset_features_20260609`).** Other instruments'
  market data (e.g. DXY/USD-basket, real yields/TIPS, VIX/risk-sentiment, other metals) MAY be ingested as READ-ONLY
  PREDICTIVE FEATURES for the XAU chain — gold does not move alone. STRICT bounds: feature-INPUT only, NEVER traded,
  NEVER an output/decision instrument, NEVER shared with another project, kept under a clearly-labelled cross-asset
  feature path. XAU remains the sole traded/output instrument; secrets + no-force-push + train==serve all unchanged.
  (Supersedes the blanket "XAUUSD only" reading of CLAUDE.md rule 6 for FEATURE inputs only — trading-scope is unchanged.)

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
  - **HARD COVERAGE RULE (user vedtak 2026-06-10 — a 36-feature silent-zero slipped through for MONTHS).** The
    liveness check MUST cover the FULL state vector of EVERY model — XGB / V10 ctx+snap / Entry-IQL (197) / V3 /
    Exit-IQL (209) — including SELF-COMPUTED features (e.g. the `attach_group_a_dip_struct_ctx_columns` 36 dip/struct
    `_v3` columns), NOT just XGB-gain + V10 ctx. ANY feature that is constant/all-zero on a shuffled sample AND not in
    `KNOWN_ALLOWED_DEAD` = HARD FAIL the build/cement, LOUD. **RESOLVED 2026-06-11 (re-audit on ACTUAL built
    values):** the 36 `dip_confirmed_{tf}_v3` / `struct_*_{tf}_v3` cols were SUSPECTED const-zero, but the full-state
    re-audit (`full_state_reaudit --detail`, shuffled sample) measured them **35/36 ALIVE** — the suspicion was WRONG.
    The coverage rule STANDS regardless: it is exactly WHY we now audit every state vector (Entry-IQL 197 + Exit-IQL 209
    + XGB + V10 ctx/snap/multi-TF), not just XGB-gain + V10 ctx. [[project_gx1_full_stack_reaudit_20260611]]
  - **MANDATE: a full feature-liveness RE-AUDIT of all 5 state vectors (actual built values, shuffled-sample variance)
    is REQUIRED before any cement and periodically — NEVER assume the allowlist is complete or that "it was checked
    once". Verify EVERY feature is FACTUALLY alive + used, every time. Do not trust memory that a feature is dead/benign
    — re-verify against the actual data.

## ONE gjeldende — artifact selection (no version roulette) [CLAUDE.md rule 8]
- **ONE truth = `PROJECT_STATE_artifacts.json`** (repo root). It names the single ACTIVE artifact per role
  (xgb / v10_entry / v3_exit / entry_iql / exit_iql, + active_variant/folds/aggregator). Edit it ONLY via an
  explicit vedtak. It SUPERSEDES `GX1_DATA/CURRENT_BUNDLES.md` (already renamed `.SUPERSEDED_SEE_PROJECT_STATE`).
- **Resolve only through the contract.** Every build/decision/serve path loads its bundle via
  `gx1_guards.load_decision_artifact` keyed on the contract — NEVER by glob, `sorted(...)[-1]`, mtime/"latest",
  or a hardcoded default path. Missing/ambiguous/PENDING ⇒ raise, never silently fall back to an on-disk vintage.
  (Known footguns flagged 2026-06-03: a V3 resolver that fell back to a stale V9 on disk, and a build-path V9
  substitution — these are the exact pattern to hunt and kill; verify none remain before trusting "can't run wrong".)
  - **Resolve-from-contract applies to AUDITING / DIAGNOSING too (2026-06-11 lesson).** When you re-audit or answer
    "what is live / is this feature dead", resolve the ACTIVE bundle's OWN data from the contract — NOT a hardcoded
    default. ENTRY and EXIT are DIFFERENT waves (entry=FASE2B_REGIME_V4, exit=FASE2B_CLEAN); `full_state_reaudit`
    hardcoded one WS2 for both → it audited the EXIT wave's forward_outcome for the ENTRY arm → a FALSE "entry
    regime-blind" verdict (the live entry IS regime-aware). Fixed: `_wave_dirs()` derives each arm's wave from the
    contract. VERIFY against the live bundle's real data before reporting a defect, every time.
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
- **NO OLD CODE LEFT BEHIND (user vedtak 2026-06-11).** The moment code is updated or replaced, the OLD
  version goes to `_legacy_disabled/` IN THE SAME COMMIT — zero orphan scripts, zero dead paths, zero
  shims importing FROM legacy (promote the implementation instead). And when LIVE-chain code changes,
  RESTART the consuming daemons in the same wave — a daemon running pre-fix code is exactly how the
  BASE34 freeze kept appending frozen rows after the fix was committed. Stale running code == old code.
- **SMART + MAXED runs — ALWAYS, before every training/build (user vedtak 2026-06-08).** Don't just raw-maximize
  CPU/GPU/RAM — design the SETUP to run as efficiently as possible. Before launching ANY train/build, ask:
  1. **Warm-start** — can I `--init-from-state-dict` from the matching cemented/prior bundle? On near-identical data
     (e.g. a data-repair/extend rebuild) a converged init cuts epochs from ~6-9 to ~1-3 (early-stop catches it). Verify
     arch match (loaded-keys missing=0/unexpected=0).
  2. **Reuse cache** — never recompute what already exists + is valid: labels/spill dirs (`--reuse-spill-dir`),
     datasets, MTF caches, candidate batches. Recompute only the dirty/extended slice.
  3. **Parallelize across idle resources** — fill the idle GPU with an independent small job while a CPU-bound step
     runs (e.g. entry-IQL retrain during V3 labeling), and run independent CPU prep while the GPU trains. Keep BOTH busy.
     **BUT RAM-HEADROOM IS A HARD CEILING (user vedtak 2026-06-10 — an OOM CRASHED the PC).** NEVER launch a parallel
     agent-Workflow or any heavy job while a build is holding tens of GB of RAM. The exit-IQL build holds ~34-42 GB
     (5M rows); stacking 4 audit agents on top tipped 47 GB → OOM → hard reboot, losing the whole run. Before launching
     anything concurrent, check `free -g` available; if a big build is resident, parallelize ONLY light read/grep work,
     and prefer reducing the build's `--sample-n-rows` to leave headroom. Watchdog the RAM (avail<1.5 GB = act).
  4. **Max throughput knobs** — `EXIT_LABEL_WORKERS≈cores-2`, `--num-workers 12`, large GPU-batch (scoring 8192), bf16+
     tf32+torch_compile. **HARD RULE — tiny IQL/MLP nets MUST train at a LARGE batch (≥4096), NEVER 256
     (user vedtak 2026-06-10).** The IQL trains via a manual in-memory minibatch loop, so batch 256 = ~13.7k
     iters/epoch, GPU ~38%, Python-loop-bound = a 16-32x throughput leak (turned a ~hour job into ~a day).
     `BUDGET_PRESETS` batch is now 4096 (was 256) + a `--batch-size` override exists — CHECK the effective
     batch in the log before walking away, and bump higher if VRAM allows. A small net is low-GPU% by nature
     → maximize SAMPLES/S (big batch), not GPU%. This applies to EVERY IQL/small-net run (exit AND entry).
  5. **Verify after launch** — snapshot GPU% / load / samples-s once warm; if a large model is CPU/dataloader-bound,
     fix (more workers / bigger batch); if a small net is low-GPU%, that's expected — don't chase GPU%.
  6. **HARD RULE — REPLAY/EVAL LOOPS MUST BE PARALLELIZED (user vedtak 2026-06-11; the gate-replay lesson).**
     Any per-candidate/per-row evaluation loop that will run more than once (gate replays, A/B arms, counter-
     factuals) MUST use a fork-after-load worker pool — per-candidate replays are embarrassingly parallel and
     Linux fork shares the big frame copy-on-write (no per-worker RAM copy). 2026-06-11: the phase6 exit-replay
     ran 5× à ~1h on 1/18 cores = ~4-5h avoidable — the SAME leak class as the IQL batch-256 rule above.
     `v12_phase6_joint_validation` honors GX1_REPLAY_WORKERS (default cores-aware; =1 forces serial). Check
     `load average` vs core count after launching ANY eval loop — 1 busy core of 18 on a >10-min job = fix it
     first, run it after. Determinism requirement: parallel output must be diff-identical to serial.
  This is a PRE-RUN discipline: think setup+efficiency first, not just "turn util up". [[feedback_check_before_build_one_truth]]
- **RETRAIN-SCOPE discipline — DON'T retrain what didn't change (2026-06-09 audit).** The audit found
  ~16h+ of avoidable pure-fit and ~45% of the 20+ transformer retrains were avoidable. Before retraining a
  TRANSFORMER ask: (1) **Warm-start by default** — 16/17 historical trainings ran COLD despite support;
  always warm-start on near-identical data (the V3 trainer now AUTO-warm-starts from the ACTIVE v3_exit
  bundle in the contract BY DEFAULT — rule-8 resolver + arch-mismatch guard; pass `--from-scratch` only
  for a real arch/contract change, or `--init-from-state-dict` to override the source). (2) **Decouple V10↔V3** —
  the cascade is one-way (XGB→V10→candidates→V3): an EXIT-side change retrains V3(+Exit-IQL) ONLY; an
  ENTRY-side change retrains V10 + regen candidates, then warm-start-REFIT V3 only if candidate drift is
  material; a full co-retrain of BOTH is justified ONLY by a SHARED-upstream change (XGB bundle or the
  `signal_bridge_v3` 41-dim SEQ contract). Never reflex "deploy together". (3) **Reward = IQL-FROZEN** —
  NEVER retrain a transformer to test a reward; relabel the reward + retrain the small IQL Q-net
  (warm-start, minutes-to-~2h), transformer FROZEN. V13 proved the from-scratch-reward path overfits AND
  is slower (Strategy-F won as an IQL-overlay with zero transformer retrain). V10 fit is ~35min (332K rows);
  V3 is the ~2h/epoch cost driver — optimize V3, treat V10 fit as ~free. [[project_gx1_retrain_cost_audit_20260609]]
- **LABEL-REWARD PRE-GATE — don't retrain on a label that doesn't separate the target OOT (user vedtak 2026-06-14).**
  Before building ANY label-reward retrain (a new reward derived from a new label set — trough-anchored, session-cond,
  vol-cond, etc.), FIRST prove the new label SEPARATES the load-bearing target on a STRICT out-of-time split (fit early
  months / test later, AND reverse). If the loser-vs-winner AUC ≈ 0.5 OOT, the retrain is FUTILE — the policy cannot
  learn what the label cannot distinguish, and a fitted filter will drop profitable trades. 2026-06-14: the
  trough-anchored entry-reward refit (to suppress the "confident-blowoff-top LONG cluster") was REFUTED exactly this way
  — trough labels overfit to GBM-AUC 0.70–0.94 in-sample but collapsed to 0.44–0.51 OOT (pre-gate ≥0.58 FAILED at 0.501),
  the "toxic" regime is net-PROFITABLE, and a label-oracle filter cost −3053 bps. The in-sample number is the trap (it
  always looks separable inside the cement's training window); the OOT split is the truth. Cheapest, run it FIRST — it
  blocks futile retrains before any compute. [[project_gx1_dd_analysis_retrain_refuted_20260614]]
- Do not run R6, freeze, promo, live, or package build without an explicit green gate.
- Keep historical artifacts as history unless an explicit selection contract marks them active.

## Running live practice
- One launcher script for the whole stack: `bash scripts/launch_live_practice.sh`.
- Idempotent — reads `*.pid` files in `GX1_DATA/reports/v12_paper_runs/`, skips anything already alive, starts only what's missing. Re-run any time to verify the stack is up.
- Starts four components together (they must ALL be running for live to track Phase 6 cement + auto-report):
  1. `v12_oanda_data_collector` — pulls M1 OHLC from OANDA practice every 60s.
  2. `v12_canonical_incremental --loop --interval 15` — appends new M1 → canonical_v3 + BASE34 prebuilts; without this, cv3 cutoff falls behind and the paper runner clips `effective_ts` to a stale bucket → live becomes a frozen replay.
  3. `v12_paper_runner` with `GX1_PURE_PHASE6=1` — disables every live-only wrapper (TIME_OF_DAY_EXIT, ADAPTIVE_MIN_ADV, REGIME, PORTFOLIO_*, LOW_CONFIDENCE, spread cap) so live = Phase 6 OOT 1:1. NOTE: CLUSTER1_RATE_LIMIT is NO LONGER disabled by PURE_PHASE6 — since 2026-06-02 it is ALWAYS ON as a live sanity-floor (v12_pipeline.py:337-345); override only via GX1_CLUSTER1_DISABLE=1 for explicit OOT-replay runs.
  4. `v12_daily_counterfactual.sh --daemon` — every hour looks for journals older than 25h that haven't been replayed yet; runs `v12_counterfactual_replay.py` on each + writes per-day "skulle/skulle ikke handlet" report to `GX1_DATA/reports/v12_paper_runs/counterfactual_reports/`. Idempotent via marker files in `.replayed_markers/`.
- Stop cleanly with `bash scripts/stop_live_practice.sh` before code edits that touch live runtime.
- **Paper-runner AUTO-RECOVER after reboot (2026-06-22):** the runner + counterfactual daemon are nohup
  (not directly systemd), so they did NOT survive reboots — the data daemons did. Now `gx1-paper-runner.service`
  (systemd --user, `Type=oneshot`, enabled in `default.target.wants/`) runs `launch_live_practice.sh` at boot,
  `After=` the data daemons + a 180s `ExecStartPre` sleep so they catch up to fresh data BEFORE the launcher's
  rule-9 preflight runs. It reuses the launcher as the ONE TRUTH (operating-point env + git-clean + 3-leg
  preflight) — NEVER duplicate that env into the unit. The preflight stays the hard gate: on a long-downtime
  data gap (forward-only daemon can't backfill a hole behind its cutoff) the launcher FATALs and the runner
  does NOT auto-start on bad data — repair the gap first (backfill M1 tape → truncate prebuilts to before the
  hole → daemon re-append; backup-first, rule 5). Logs: `…/logs/auto_recover.log`. NOTE: `systemctl --user`
  needs the dbus user session (`/run/user/UID/bus`), which WSL doesn't always start at boot — if it's down,
  enable via the `default.target.wants/` symlink directly (bus-independent; the manager reads it at boot).
- **Continuous-learning loop (ladder wave 2026-06-12):** `gx1-nightly-learning.timer` (systemd --user,
  03:30 UTC) runs `scripts/gx1_nightly_learning.sh` — per-trade verdict accumulation
  (`counterfactual_reports/trade_verdicts_*.jsonl` → `nightly_learning/regret_dataset.parquet`), the
  CANONICAL-TAPE FRESHENER (writes M1+M5 canonical tapes nightly: OANDA-history backfill + downsample,
  idempotent, fail-loud on skipped fetch chunks), rolling matured live-regret replay buffers — ENTRY
  (`build_online_replay_buffer`, cement-M5 label convention + SYM parity, D-8..D-2) and EXIT
  (`build_online_exit_replay_buffer`, per-(trade, M1-bar) 209-dim transitions, dataset-only) — and the
  rule-9 KS distribution-drift leg (`feature_liveness --distribution-drift`, ADVISORY — flags a retrain
  vedtak, never retrains; reference = `drift_reference_v1.parquet` in the ACTIVE entry bundle,
  name-aligned). The REFIT leg is armed ONLY by a standing vedtak id in
  `GX1_DATA/config/nightly_refit_standing_vedtak.txt` (rule 3 — ARMED 2026-06-12:
  `nightly_iql_refit_standing_v1`); it runs a 3-fold warm-start with the ANTI-FORGETTING cement mix
  (`cement_replay_sample_v1.parquet` in the ACTIVE bundle, `--mix-cement`), writes a PENDING candidate
  under `reports/online_iql/warmstart_<ts>/`, shadow-scores it on D-1 (out-of-sample only), runs
  `scripts/gx1_candidate_gate.sh --quick` (evidence json under `nightly_learning/candidate_gates/`;
  FULL mode incl. the decisive volbal-baseline comparison is required before any flip), and rotates the
  in-process shadow (`GX1_DATA/config/shadow_bundle_dir.txt`) ONLY on gate PASS — promote stays a manual
  contract flip (rule 8). **In-process shadow:** when `shadow_bundle_dir.txt` names a candidate, the
  paper runner loads a second Entry-IQL adapter (fail-SAFE, auto-resolves variant/fold from the
  candidate's own checkpoints) that scores every poll through the live `predict()` path and journals
  `shadow_action`/`shadow_q_per_action`/`shadow_agrees_with_live` — affects nothing, picked up at runner
  restart, mid-run disablement journaled as `shadow_disabled_reason`. Track B variant-shadow defaults to
  `--variants auto` (enumerates the contract-resolved bundle's checkpoints; a bundle flip can no longer
  silently empty the shadow) and honors `GX1_SHADOW_BUNDLE_DIR` for shadowing a PENDING candidate.
- The data daemons (collector + canonical_incremental) run under **systemd --user** (`gx1-collector.service`, `gx1-canonical-incremental.service`) and log to `/home/andre2/GX1_DATA/reports/v12_paper_runs/logs/{collector,canonical_incremental}.log`. The incremental unit has a drop-in (`~/.config/systemd/user/gx1-canonical-incremental.service.d/ctx_env.conf`) pinning `GX1_TREND_REGIME_FROM_D1=1` — REQUIRED for the BASE34 ctx recompute (without it trend_regime_id degenerates to constant 1; 2026-06-11 freeze fix) — AND `GX1_REGIME_V4=1` (2026-06-13 114-col cutover) — REQUIRED so the daemon emits the 52 REGIME_V4 BASE34 cols; without it they carry-forward FROZEN on append (the 2026-05-25 freeze class). The daemon pairs this with the one-truth `htf_features.attach_default_regime_v4_v2_scalars` (per-TF V2 inputs computed BEFORE `augment_canonical_v3`) + a full-history D1-EWM recompute in the cv3 append (`compute_d1_features` over full cv3, so `d1_rsi14`/`d1_ema_slope_20` converge — was 30-day warmup). If you ever restart the daemon, KEEP both flags (commit 3e115762). The `launch_live_practice.sh` nohup fallback (which would log to `/tmp/gx1_live_practice/`) is NOT the live source — prefer the systemd units (`systemctl --user status gx1-collector gx1-canonical-incremental`).

## Git & secrets
- Never amend live commits. Never force push.
- Never commit secrets (`.env`, credentials).
