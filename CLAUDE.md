# ACTIVE ENTRY/EXIT SUPER AI BOT OVERRIDE - 2026-07-08 (SMART CHAIN PROMOTED)

The Entry/Exit AI workstream is governed by
`docs/ACTIVE_SUPER_AI_BOT_GOAL_20260702.md` (see its STATUS 2026-07-08 section),
`/home/andre2/GX1_DATA/HANDOVER_2026_07_08_SMART_CHAIN_PROMOTED.md` (baton pass)
and `PROJECT_STATE_artifacts.json` (the ONE selection truth; promotion commit
d98bc61e). The 2026-07-02 smoke-wave instructions are COMPLETED history — the
smart chain qualified the entire ladder and was PROMOTED. Legacy live-practice
instructions are dead: the legacy entry bundles are PHYSICALLY GONE
(20260707 delete-incident) and that chain is RETIRED.

Active state (every flip by explicit user vedtak):
- v10_entry = smart_seq520 cand#4 ACTIVE (vedtak
  SMART_JOINT_POLICY_PROMOTION_20260708, commit d98bc61e). Qualified the full
  ladder: pin-aligned bundle audit PASS (all hash checks), calibrated
  (direction NLL 1.87->1.02, path corr +0.251), replay-readiness READY on both
  identities. Pinned operating point: sessions US+OVERLAP, edge_score threshold
  0.16176772117614746 (top-20% of US+OVERLAP VAL Q4-2025), fill = M1 open at
  prediction-bar close (T+5), max_trades=3. Smart chain v1 is CANDIDATE-POLICY
  ONLY — no entry-IQL layer.
- xgb = surviving May-2026 CPU base80 bundle ACTIVE
  (`models/xgb_v7_base80_20260526_cpu_PROMOTED_20260708`) — role is V3-exit-
  bridge input ONLY; the smart entry runs a NEUTRAL bridge.
- exit_iql = `runs/FASE2B_CLEAN_20260608/exit_iql_deferral_20260707` ACTIVE
  (vedtak EXIT_IQL_DEFERRAL_PROMOTION_20260707, commit 8e252246). Cap-3 phase6:
  return/DD 14.50 (W2026) / 23.31 (W2025) vs re-based baseline 9.52/11.05, with
  LOWER account-DD. Serving REQUIRES GX1_STRONG_HOLD_QADV=-66.5 +
  GX1_STRATEGY_F_DEFER_CAP_BARS=240 — pinned in the contract's
  exit_iql.operating_point.live_env; `scripts/gx1_exit_env_pin.sh` is the ONE
  truth every gate/replay must eval before running.
- entry_iql = RETIRED. The grid-proxy student was honestly refuted (all knobs
  exhausted 2026-07-07); the REAL Q-net student is research-PENDING
  (`runs/entry_iql_research/real_student_20260707/` — beats candidate on PnL
  but fails PF/DD bounds). Reopens only via its own gates + explicit vedtak.
- JOINT policy replay-proven (`reports/joint_smart_policy_replay_20260708/`,
  pre-registered, one run): 3506 trades jan-jul 2026, per-trade EV 74.69 bps,
  win 0.944; cap-3 account +52,867 bps, maxDD 805 bps, return/DD per month
  15.5 / 11.4 / 17.4 / 14.1 / 21.1 / 15.3 — every month above legacy ~9-11.

Current allowed path (SERVING WAVE — in flight):
- build the live serving path for the promoted smart chain: live per-M5 520-dim
  state-builder + smart-entry adapter + train==serve parity gate + runner
  integration. Under construction in `gx1/execution` — extend that work, never
  fork a parallel serving path.
- parity gate: live serve output must be replay-identical on the same bars
  before anything downstream opens.
- after parity PASS: rule-9 three-leg preflight (live-tail freeze hard-fail /
  continuity-gap hard-fail / KS-drift advisory) on the live data path.
- demo/paper launch ONLY after parity gate PASS + preflight + an explicit user
  launch vedtak.

Current blocked path:
- no live/demo/paper order placement until the serving wave lands, the parity
  gate PASSes, preflight passes and an explicit launch vedtak exists.
- no legacy live/practice launch: the legacy entry chain is RETIRED and its
  bundles physically gone — do not "repair" `launch_live_practice.sh` back onto
  the legacy chain; the serving wave replaces it.
- no entry retraining for direction — the information ceiling stands; the
  promoted candidate-policy v1 is the selection truth.
- parked/refuted tracks (do not reopen without new data or explicit vedtak):
  M1 mid-trade exit timing (3x independent statistical null at n=218
  2026-episodes — needs MORE LIVE EPISODES, not more retrains); IQL grid-proxy
  student (refuted); hold-horizon label head (OOT pregate FAIL — head stays
  inactive); full-history dense exit substrate (entry-IQL provenance is
  2026-bound + hash-pinned — top-20% selection not provable pre-2026).

Hard lessons codified 2026-07-08 (apply always):
- an AUC-label-pregate proves label SEPARABILITY only, never replay value; any
  timing/label track additionally requires a REPLAY-SIMULATED pregate
  (first-trigger dynamics vs the actual policy) before any training.
- delete-executors HARD-FAIL on unresolvable exclusion paths, and rule-5
  dry-run inventories carry FULL paths, never '...'-abbreviated — the 20260707
  delete-incident destroyed 6 contract-referenced legacy rollback bundles
  because abbreviated exclusion paths silently failed to resolve.
- the exit operating-point env comes from the contract via
  `scripts/gx1_exit_env_pin.sh` — never a per-script copy. Nightly true-netcap
  numbers from BEFORE 2026-07-07 measured a DIFFERENT exit policy (unpinned
  env) and are decision-invalid.
- heavy jobs must preserve RAM headroom and use capped runners where required.

# GX1 — operational constitution (read every session)

These are the HARD rules. They are operational, not architectural — they apply to every
session, every turn. Make a violation impossible or loud, never "remember to be careful".
Architecture, data-flow, and the detailed guardrails live in @AGENTS.md — read it before acting.

## OPERATING MODE — DRIVE TO COMPLETION (user vedtak 2026-06-05)
Default to AUTONOMOUS execution. When I authorize a goal/wave (a `--vedtak`, or "kjør alt ferdig /
fortsett til du er fornøyd"), execute every step of it — builds, runs, retrains, data-prep, cleanup —
to completion WITHOUT per-step confirmation. Verify each step yourself; report at MILESTONES, not before
each step. One wave `--vedtak` covers every retrain in that wave (rule 3). Stop ONLY for (a) a genuine
hard blocker you cannot resolve after actually trying, or (b) a catastrophe-floor breach (rules 2/4/6:
git-clean-before-run, manifest, XAU-only / no-secrets / no-force-push). Continue until DONE + verified.
Asking me to confirm every little thing creates misery — DON'T. (Protected-core edits are no longer
marker-gated as of 2026-06-05 — the hook warns, doesn't block; edit the live chain deliberately per
rule 1, no per-edit `touch`. Reversible-first cleanup per rule 5 needs no per-item ask within an authorized wave.)

## ALWAYS BUILD, NEVER REMOVE — FIX AND IMPLEMENT (user vedtak 2026-06-13, MUST ALWAYS BE FOLLOWED)
When two artifacts/schemas/feature-sets differ, the answer is NEVER to DROP capability to match the
lesser one — it is to UPGRADE the lesser one UP to the richer one. We drop NOTHING. A richer schema
(more features / more regime cols / more capability) is the target; bring everything else up to it,
fix the wiring, implement the missing side. Example (the mistake that birthed this rule): a rebuilt
BASE34 had 114 cols WITH the REGIME_V4 block; the live serve had 62 WITHOUT it. The WRONG move is
"rebuild regime-OFF to match live's 62"; the RIGHT move is "keep the 114 WITH regime and upgrade the
live serve + daemon to the 114-col regime-aware schema." Removing features to make a swap easier is
forbidden. If a model/serve path lacks a feature the build has, that is a GAP to CLOSE (build it in),
never a reason to delete the feature. Always build, fix, implement — never amputate.

1. **Protected core — edit DELIBERATELY (hard marker-gate removed 2026-06-05, user vedtak).** The live
   chain / SACRED transformer contracts — `gx1/execution/`, `gx1/contracts/`, `gx1/exits/contracts/`,
   `gx1/models/entry_v10/`, `gx1/core/` — are no longer marker-gated (the per-edit `touch` friction was
   killing the workflow); the PreToolUse hook now WARNS instead of blocks. The discipline is UNCHANGED:
   verify in-use, ONE truth, minimal change, train==serve, and NEVER coarsen the M1 exit grid (that edit
   is still HARD-blocked). The gate is gone; the care is not — treat every live-chain edit as a real change.

2. **Git clean before any run.** Never start a train / retrain / backtest / Phase-6 / live launch
   while `git status --short` is non-empty. A dirty tree means we don't know what we're running.
   No exceptions.

3. **Never auto-retrain.** Every retrain requires an explicit `--vedtak <id>`. The `gx1_guards`
   are fail-closed; do not route around them.

4. **Manifest before every run.** Always produce and log a run-manifest first (`/run-experiment`):
   commit hash, config path+hash, dataset path+rows+hash, checkpoint, seed, feature-set version.
   No silent defaults — an explicit `--config` is required; missing config is a hard error, not a
   guessed default.

5. **Never delete artifacts without confirmation.** No deletes under `GX1_DATA/` or `runs/`.
   Destructive cleanup is always backup → inventory → dry-run → my confirmation → delete, in that
   order.

6. **XAUUSD only. No secrets. No force-push.** Never introduce another instrument. Never commit
   `.env`/credentials. Never `git push --force` or `git reset --hard` shared history.

7. **Extend, don't fork; check before you build.** Change the existing component; new files only
   for a genuinely-new shared one-truth helper. Before creating a file, name the existing one you
   considered extending and why it didn't fit. Keep ONE truth, fail-closed, minimal change.

8. **ONE gjeldende — single selection truth, no version roulette (user vedtak 2026-06-05).** Exactly
   ONE artifact is active per role (xgb / v10_entry / v3_exit / entry_iql / exit_iql), named in the ONE
   selection contract `PROJECT_STATE_artifacts.json`. Build / decision / serve code resolves bundles ONLY
   through that contract (`gx1_guards.load_decision_artifact`) — NEVER by glob, "latest", mtime, or a
   hardcoded/default path. Missing or ambiguous = hard error, never a guessed fallback. A new artifact is
   PENDING until it passes gates and I flip the contract; the live-active bundle is never deleted while
   active. **AUTO-PARK on supersede (user vedtak 2026-06-06): the moment a new version supersedes an old
   one (e.g. V7→V8 exit contract), AUTOMATICALLY quarantine the superseded build-artifacts — reversible
   move to `runs/_SUPERSEDED_<date>/`, WITHOUT being asked.** Superseded DATASETS / intermediate builds /
   non-active dupes park immediately (rollback re-activates the BUNDLE, not the dataset, so datasets are
   safe to park once a newer dataset exists); the still-live-ACTIVE BUNDLE parks automatically at the
   cement-flip that deactivates it — never before (it is the rollback). Quarantine→delete via rule 5
   (backup→inventory→dry-run→my confirm) so we never drown in v1/v2/v3… The goal is that running the wrong
   version is IMPOSSIBLE (a fail-closed resolver refusing to guess), not "remember to pick the right one".

9. **NOTHING ignored — feature-liveness is AUTO-checked every run, NEVER hand-verified (user vedtak
   2026-06-06).** Every input feature + dependency the chain consumes MUST be alive (non-constant on a
   SHUFFLED sample) or on the documented `gx1.audit.feature_liveness.KNOWN_ALLOWED_DEAD` allowlist. The
   ONE-TRUTH cross-chain check `gx1.audit.feature_liveness` runs AUTOMATICALLY + fail-closed: the XGB-gain
   gate at XGB-retrain post-export, the V10 ctx/snap/multi-TF (all 5 TFs alive + DISTINCT) check at
   V10-retrain post-export, the zeroed/constant-hand-off guard in the Entry/Exit-IQL builds, and
   `python -m gx1.audit.feature_liveness --strict …` before any cement. A NEW dead/ignored feature (or a
   broken/duplicated TF, or a zeroed hand-off) = a silent-ignore regression → the build/retrain/cement
   FAILS LOUD; fix the wiring or add to the allowlist with a documented reason. NEVER hand-verify "are all
   features used" again — the chain refuses to run-wrong. **LIVE-TAIL leg (user vedtak 2026-06-11): the
   check also covers the LIVE prebuilt tails** — `--live-tail` scans cv3+BASE34 for the freeze signature
   (was-varying column now constant; the 2026-05-25 BASE34 copy-forward freeze lived 17 days while every
   training-side audit was green). Runs AUTO: hard-fail preflight in `launch_live_practice.sh` + hourly
   ERROR-loud self-check in the canonical_incremental daemon. Allowlist `LIVE_TAIL_ALLOWED_CONST`
   (documented reasons only). **EVERY (re)start preflight = THREE legs (user-direktiv 2026-06-12):**
   (1) live-tail FREEZE — hard fail; (2) CONTINUITY/gap-sjekk mot helg/pause/helligdager/KNOWN_DATA_GAPS —
   fresh UNKNOWN gap = hard fail (hull i historikken blokkerer oppstart); (3) KS DISTRIBUTION-DRIFT —
   siste 7d live-states vs ACTIVE-bundlets `drift_reference_v1.parquet`, ADVISORY-loud (markedsdrift
   flagger et retrain-VEDTAK, blokkerer aldri en launch — rule 3; en bot som nekter å starte i nytt
   regime er feil). Drift-referansen regenereres ved hvert cement (`--write-drift-reference`).

@AGENTS.md
