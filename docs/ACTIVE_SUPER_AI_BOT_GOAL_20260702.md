# Active Super AI Bot Goal - 2026-07-02

Status: active objective and roadmap. This document supersedes older Entry-only,
V9/V10/XGB, SNIPER and historical live-practice notes as the high-level goal
for current Entry/Exit AI work. Detailed gates remain in
`docs/ENTRY_FOUNDATION_AUDIT_20260628.md` and
`docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md`.

## STATUS 2026-07-16 - XAU DIRECTION REPAIR ACTIVE

The active work is no longer the 2026-07-08 serving wave. That state is
historical context only and is not launch-valid for the current repair.

Current source of truth for live work:
- `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`
- `bash scripts/gx1_handover.sh`
- `AGENTS.md` active XAU direction-repair override

Current position:
- Fresh 20260716 XAU rebuild, smoke dataset, and report-only readiness gates are
  green.
- Smart smoke training is bound to the fresh XAU contract and fails closed on
  stale/missing paths.
- No fresh XAU transformer bundle has passed hard direction-slice and
  class-balance gates.
- Latest bounded smart smoke attempts remained hard-red after source-level
  repairs including independent public FLAT head, side-gradient staging, and
  public trade/FLAT pairwise margin. No candidate bundle was produced.
- Candidate training, replay, Entry-IQL, shadow, live, and promotion are closed.

Active roadmap from here:
1. Keep disk/RAM/process hygiene green before touching heavy jobs. Clean up
   aborted manifests/tmp/runs after extracting evidence.
2. Hunt code and contract mismatches before another run: stale paths, readiness
   fields missing from audit, report/latest races, and fallback behavior.
3. Require fresh source identity in every smart XAU gate: post-rebuild readiness,
   smoke-readiness, trainability, future train argv and output bundle root must
   point at the same fresh rebuild. Newer red reports invalidate older READY.
4. Diagnose or repair the Entry Transformer public trade-vs-FLAT hard decision
   surface. Do not move to Entry-IQL and do not rerun unchanged hard-red recipes.
5. Do not allow smart direction-repair recipe overrides except under a separate
   audited sweep contract with exact values carried through readiness/audit.
6. After a small source repair, run focused tests plus smart smoke readiness,
   trainability readiness, and train enablement from a clean git tree.
7. Run at most one bounded fresh XAU smoke for that new formulation, with
   hard-red stop discipline.
8. Open candidate/replay/IQL only after a fresh XAU transformer bundle passes
   hard direction-slice and class-balance gates.

## STATUS 2026-07-08 - HISTORICAL SMART JOINT POLICY CONTEXT

The ladder in this document has been executed through step 9's replay proof.
The sections below ("Current Position", "Roadmap" steps 1-6) describe the
2026-07-02 smoke-wave starting point and are COMPLETED history; the contract
`PROJECT_STATE_artifacts.json` (promotion commit d98bc61e) is the selection
truth. Baton pass with full evidence paths and rollback recipes:
`/home/andre2/GX1_DATA/HANDOVER_2026_07_08_SMART_CHAIN_PROMOTED.md`.

What is proven and ACTIVE (each flip by explicit user vedtak):

- Entry: smart_seq520 cand#4 qualified the whole ladder — pin-aligned bundle
  audit PASS, calibrated (direction NLL 1.87 -> 1.02, path corr +0.251),
  replay-readiness READY on both identities — and is ACTIVE as v10_entry
  (vedtak SMART_JOINT_POLICY_PROMOTION_20260708). Pinned operating point:
  US+OVERLAP, edge_score threshold 0.16176772117614746 (top-20% of US+OVERLAP
  VAL Q4-2025), M1-open fill at T+5, max_trades=3. Smart chain v1 runs
  CANDIDATE-POLICY ONLY (no entry-IQL layer; role RETIRED).
- Exit: exit_iql_deferral_20260707 ACTIVE (vedtak
  EXIT_IQL_DEFERRAL_PROMOTION_20260707, commit 8e252246) — cap-3 return/DD
  14.50/23.31 vs re-based baseline 9.52/11.05 with lower account-DD. Serving
  requires GX1_STRONG_HOLD_QADV=-66.5 + GX1_STRATEGY_F_DEFER_CAP_BARS=240
  (contract live_env; pin helper `scripts/gx1_exit_env_pin.sh`).
- Joint policy replay-proven (pre-registered, one run):
  `reports/joint_smart_policy_replay_20260708/` — 3506 trades jan-jul 2026,
  per-trade EV 74.69 bps, win 0.944; cap-3 account +52,867 bps, maxDD 805 bps,
  return/DD per month 15.5 / 11.4 / 17.4 / 14.1 / 21.1 / 15.3 (every month
  above the legacy ~9-11 reference).
- XGB: May-2026 CPU base80 bundle ACTIVE for the V3-exit-bridge only; the
  smart entry runs a neutral bridge.

Refuted/parked this wave (reopen only with new data + explicit vedtak):

- M1 mid-trade exit timing: three independent statistical nulls at n=218
  2026-episodes (`reports/exit_parity_wave_20260704_evidence/
  m1_timing_train_20260707/RESULT.md`) — needs more live episodes, not
  retrains. Lesson: label-AUC pregate proves separability, not replay value;
  a replay-simulated pregate is now required before any timing train.
- Entry-IQL grid-proxy student: honestly refuted (cannot beat the calibrated
  candidate). REAL Q-net student research-PENDING
  (`runs/entry_iql_research/real_student_20260707/`, beats candidate PnL but
  fails PF/DD bounds).
- hold_horizon label head: OOT pregate FAIL (`reports/
  cand4_julyext_evidence_20260705/fixwave_20260707/hold_label_pregate/
  RESULT.md`) — head stays inactive, fail-loud.
- Full-history dense exit substrate: blocked on provenance (entry-IQL
  2026-bound + hash-pinned; top-20% selection not provable pre-2026).

## Historical Roadmap From 2026-07-08

1. SERVING WAVE (in flight, `gx1/execution`): live per-M5 520-dim
   state-builder + smart-entry adapter + runner integration. Extend the
   in-flight work; never fork a parallel serving path.
2. TRAIN==SERVE PARITY GATE: live serve output replay-identical on the same
   bars (hard gate; nothing downstream opens before PASS).
3. RULE-9 PREFLIGHT on the live data path (live-tail freeze hard-fail /
   continuity-gap hard-fail / KS-drift advisory).
4. DEMO/PAPER LAUNCH — only after 2+3 PASS and an explicit user launch vedtak.
5. DATA ACCUMULATION under demo: live episodes for the parked M1-timing track,
   live selected-trade distribution vs replay, deferral-exit liveness.
6. ROUND-3 RESEARCH TRACKS (each its own pregate + vedtak): real Q-net student
   vs its PF/DD bounds; M1 timing re-attempt ONLY once enough live episodes
   exist AND a replay-simulated pregate passes; threshold recalibration on a
   fresh val window (explicit vedtak, per the contract's recalibration
   policy).

## Goal

Build a fully automated XAUUSD policy where all Entry and Exit inputs cooperate
as one calibrated multi-timeframe market-state language.

The bot must:

- enter long or short only when the combined Entry evidence is high quality;
- skip weak or dangerous market states instead of forcing a direction;
- exit near maximum profit opportunity, not merely after a fixed hold;
- preserve the reason for entry into Exit so profit capture can depend on the
  same structure, liquidity, momentum, trend, volatility, session and regime
  context that created the trade;
- prove edge with replay net PnL, drawdown, MAE, bad-path, path quality,
  selected-tail direction quality and session/regime/side robustness.

Raw direction accuracy is only a diagnostic. A model with attractive broad
accuracy but weak replay, weak FLAT calibration, bad selected-tail direction or
hidden session/regime failure is not a candidate.

## Current Position

The foundation is no longer just SMC. The active research surface is:

- `foundation_seq146`: activated canonical foundation and six trainable
  specialists.
- `challenger_seq215`: audited eight-specialist challenger with chart geometry
  and price-action/candle specialists.
- `smart_seq520_candidate`: structurally ready smart candidate with 520
  seq/snap signals, 142 continuous context fields, 5 categorical context
  embeddings and 305 smart-layer mechanism features.
- Feature harmony: current inventory accounts for all active/generated Entry
  inputs; routed or explicitly excluded fields must remain at zero unmapped.
- Smart trainability: structurally ready, but not yet proven by a new capped
  smart smoke train.

The latest readiness state is:

- worktree: `PASS_CLEAN_GIT`;
- smart smoke readiness: ready for review;
- smart trainability readiness: ready for review;
- smart direction benchmark: not ready;
- known failure mode: smart520 underpredicts FLAT and overcalls LONG/SHORT;
- latest repair: main direction loss, MTF direction auxiliary loss, checkpoint
  guard and bundle audit now preserve the smart FLAT/class-balance repair.

## Feature Language

The roughly 200 old foundation features are not replaced by smart features.
They remain the source language. Smart features are calibrated summaries on top
of that same source language and must keep provenance.

Every active input must be one of:

- routed to a specialist or smart mechanism family;
- explicitly excluded with a recorded reason;
- proven live, finite, hashed and non-collapsed;
- carried forward into Exit state when it is needed to manage trades opened by
  that Entry logic.

Primary mechanism families:

- structure/swing: HH/HL/LH/LL, BOS/CHoCH, impulse, pullback;
- SMC/liquidity: sweeps, reclaim, false breakouts, premium/discount, levels;
- trend/EMA: EMA stack, price-vs-EMA, MTF trend pressure;
- momentum/flow: returns, CLV, impulse follow-through, exhaustion;
- volatility/compression: ATR, squeeze, expansion/release, spread;
- session/regime: Asia/EU/US/overlap, regime stack, session interactions;
- chart geometry: trendlines, channels, Fibonacci, support/resistance, flags;
- price action/candles: body/wick, doji/hammer/engulfing/inside/outside style
  pressure;
- MTF confluence: M5/M15/H1/H4/D1 agreement and divergence;
- Entry-to-Exit state: the exact Entry context and specialist-gate weights Exit
  needs to learn profit capture.

## Model Stack

The intended AI stack is cooperative, not a set of isolated bots:

- Entry Transformer: evidence layer, reads sequence/snap/context and specialist
  fusion.
- Entry IQL: entry policy layer, allowed only after replay evidence proves the
  Transformer candidate has tradable edge.
- Exit Transformer: exit-timing evidence layer, trained from exact Entry replay
  traces and Entry-bound per-bar state.
- Exit IQL: exit policy layer, allowed only after Exit Transformer/replay gates
  prove it improves realized profit capture.
- Challenger models: PatchTST, TSMixer and later TimesNet/TFT may challenge the
  specialist Transformer only under the same replay, slice, calibration and
  artifact-identity gates.
- Auxiliary smart models may exist only as audited heads, frozen features or
  gated specialists inside the same contract. They must not become independent
  uncontrolled entry or exit bots.

## Current Fail-Closed Rules

- Do not train on stale foundation.
- Do not start smoke/candidate training without clean git, required readiness
  and explicit matching vedtak.
- Do not start replay, IQL distillation, shadow, live or promotion from weak or
  stale artifacts.
- Do not hide weak behavior behind broad averages. Always check session,
  regime, side, direction, bad-path, selected-tail and tail-loss slices.
- Do not remove old features just because smart summaries exist. Removal needs
  an exact ablation plus replay/slice proof that edge improves.
- If a gate lacks provenance, hashes, liveness, exact contract, active/blocked
  head parity or specialist gate liveness, fix the gate before moving forward.
- RAM is a hard operating constraint. Heavy train/replay jobs must use capped
  runners, `num_workers=0` where required, and live memory checks.

## Roadmap

1. Smart smoke proof
   - Run only after explicit SMART/SEQ520 smoke vedtak and clean git.
   - Use repaired direction-balance recipe:
     `ENTRY_PRED_BALANCE_ALPHA=0.20`,
     `ENTRY_PRED_BALANCE_CLASS_WEIGHTS=1.0,1.0,4.0`,
     `GX1_V10_CKPT_MONITOR=dir_acc` and active class-balance checkpoint guard.
   - Require MTF direction auxiliary head to prove
     `mtf_dir_aux_uses_direction_balance_repair=true`.
   - Tunable within the same SMART/SEQ520 vedtak: early-stop patience via
     `--early-stop-patience <n>` (wrapper default 1) and stronger FLAT press via
     explicit `ENTRY_FOUNDATION_SMOKE_PRED_BALANCE_*` env overrides — overrides
     are loud-logged and recorded in the pre-train manifest. Smoke dataset size
     is enlarged only via the sanctioned
     `smart-post-rebuild-refresh --apply --vedtak <id> [--train-rows <n>]
     [--val-rows <n>] [--test-rows <n>]` gate followed by a regenerated
     smoke-manifest.

2. Post-smoke bundle audit
   - Edge scope (vedtak SMART_SEQ520_smoke_wave_20260702): the smoke audit runs
     `--edge-test-scope smoke` — whole-split majority-beat and the global
     direction-distribution contract stay HARD on every split, while
     per-slice direction diagnostics and path/bad_path head-sign checks on the
     strict-OOT TEST split are loud advisories at smoke stage. They remain
     HARD (`strict`, the default) at candidate stage (steps 4-5), where 2026
     data enters training.
   - Prove strict load, finite forward, exact active heads, blocked
     `hold_horizon`, exact specialist contract, non-collapsed specialist gate
     liveness, path calibration, tail-direction recipe and direction-balance
     repair.
   - Fail closed unless smart direction/class balance improves enough to stop
     the current benchmark regressions.

3. Smart candidate training
   - Allowed only after smart candidate-readiness is green.
   - Preserve the exact smart520 specialist contract and smoke provenance.
   - Keep XGB bridge neutral/diagnostic unless a separate gate proves it helps.

4. Selective-edge and no-XGB ablation
   - Measure selected trades, not just full-sample accuracy.
   - Require no-XGB ablation and slice diagnostics.

5. Candidate replay evidence
   - Use explicit trades with policy id, side, probabilities, score, PnL, MFE,
     MAE and held bars.
   - Prove net PnL, profit factor, drawdown, max loss, selected-tail direction,
     bad-path avoidance and month/session/regime/side robustness.

6. Entry IQL distillation
   - Open only after replay-readiness is green.
   - Distill from replay rewards tied to the exact candidate bundle and trade
     logs.
   - IQL replay and comparison must beat the candidate without worsening
     drawdown, max loss, profit factor or negative-month behavior.

7. Entry-to-Exit handoff
   - Materialize exact Entry-bound per-bar state from replay trades.
   - Preserve smart520 snapshot fields and specialist-gate weights needed by
     Exit.
   - No Exit training if the Entry reason-for-trade is missing from Exit state.

8. Exit Transformer and Exit IQL
   - Train only after Exit feature alignment, reconstruction, state/reward,
     split/leakage, architecture, training-plan, wrapper, pretrain-manifest,
     slice robustness, train-execution and post-train audit gates pass.
   - Optimize profit capture: MFE capture, giveback, MAE, drawdown, terminal
     reward, side/session/regime/tail slices.

9. Promotion review
   - Promotion, shadow and live remain closed until the full Entry + Entry IQL
     + Exit + Exit IQL policy is replay-proven as one live-equivalent system.

## Definition Of Done

The bot is not done when a training metric improves. It is done only when a
single artifact-identified policy proves:

- all active inputs are routed/excluded, live, finite, hashed and non-collapsed;
- Entry evidence improves selected replay edge versus baselines;
- Entry IQL beats the Entry candidate in replay;
- Exit Transformer/IQL improve profit capture from the exact Entry traces;
- no broad average hides failure in side, session, regime, volatility, spread,
  bad-path or tail slices;
- train/replay/live parity is proven;
- promotion review explicitly opens shadow/live, and until then
  `promotion_shadow_live_allowed=false`.
