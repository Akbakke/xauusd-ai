#!/usr/bin/env bash
# GX1 HANDOVER — print the CURRENT gjeldende state for a fresh session.
#
# Re-runnable orientation: reads PROJECT_STATE_artifacts.json (the ONE selection truth) live, checks the
# stack, and prints the operating point + edge-buildout state + verify/rollback/next steps.
# Resolve bundles ONLY via gx1_guards.load_decision_artifact — never glob/latest/mtime.
#
# Usage:  bash scripts/gx1_handover.sh
set -euo pipefail
REPO=/home/andre2/src/GX1_ENGINE
DATA=/home/andre2/GX1_DATA
PY=$REPO/.venv/bin/python
CONTRACT=$REPO/PROJECT_STATE_artifacts.json
PAPER=$DATA/reports/v12_paper_runs
cd "$REPO"

if [[ "${GX1_ALLOW_LEGACY_HANDOVER:-}" != "20260627_ALLOW_LEGACY_HANDOVER" ]]; then
  bar(){ printf '%s\n' "------------------------------------------------------------------------"; }
  echo
  bar
  echo "  GX1 HANDOVER - active Entry foundation seq146 @ $(date -u '+%Y-%m-%d %H:%M:%SZ')"
  echo "  Canonical docs:"
  echo "    docs/ENTRY_FOUNDATION_AUDIT_20260628.md"
  echo "    docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md"
  echo "  Control surface:"
  echo "    scripts/entry_next_edge_control.sh"
  bar
  echo
  echo "Verification:"
  VERIFY_ERR=$(mktemp)
  if "$PY" -m gx1.scripts.verify_entry_foundation_state_v1 --quiet 2>"$VERIFY_ERR"; then
    echo "  foundation verify: PASS"
  else
    echo "  foundation verify: FAIL (see feature/target/specialist readiness artifacts)"
    LAST_VERIFY_ERROR=$(grep -E "RuntimeError:|Error:" "$VERIFY_ERR" | tail -n 1 || true)
    if [[ -n "${LAST_VERIFY_ERROR:-}" ]]; then
      echo "  foundation verify error: $LAST_VERIFY_ERROR"
    fi
  fi
  rm -f "$VERIFY_ERR"
  if [[ "${GX1_HANDOVER_SKIP_GUARDRAILS:-}" == "1" ]]; then
    echo "  foundation guardrails: SKIPPED_BY_GUARDRAIL_TEST"
  else
    "$REPO/scripts/entry_next_edge_control.sh" foundation-guardrails --quiet
    echo "  foundation guardrails: PASS"
  fi
  if [[ "${GX1_HANDOVER_SKIP_TRAIN_READINESS:-}" == "1" ]]; then
    echo "  train-readiness: SKIPPED_BY_GUARDRAIL_TEST"
  else
    "$REPO/scripts/entry_next_edge_control.sh" train-readiness --quiet --no-fail-on-not-ready
    "$PY" - <<'PY'
import json
from pathlib import Path

path = Path("/home/andre2/GX1_DATA/reports/entry_training_readiness_20260628_v1/ENTRY_TRAINING_READINESS_latest.json")
report = json.loads(path.read_text(encoding="utf-8"))
print(f"  train-readiness: {report.get('decision')}")
print(f"  failures: {len(report.get('failures') or [])}")
print(f"  foundation contract ready: {report.get('foundation_contract_ready_for_smoke')}")
print(f"  foundation activation required: {report.get('foundation_activation_required_before_smoke')}")
print(f"  foundation activation apply required: {report.get('foundation_activation_apply_required_before_smoke')}")
print(f"  foundation post-apply required: {report.get('foundation_activation_post_apply_required_before_smoke')}")
print(f"  execution blockers: {len(report.get('execution_blockers') or [])}")
hygiene_raw = (report.get("artifacts") or {}).get("worktree_hygiene")
hygiene_path = Path(hygiene_raw) if hygiene_raw else None
if hygiene_path and hygiene_path.exists():
    hygiene = json.loads(hygiene_path.read_text(encoding="utf-8"))
    print(f"  worktree hygiene: {hygiene.get('decision')} dirty={hygiene.get('dirty_count')}")
    print(f"  foundation cleanup dirty: {hygiene.get('foundation_cleanup_dirty_count')}")
    print(f"  review-before-stage dirty: {hygiene.get('review_before_stage_dirty_count')}")
    stage_summary = hygiene.get("foundation_stage_summary") or {}
    hold_summary = hygiene.get("review_hold_summary") or {}
    print(f"  stage bytes/lines: {stage_summary.get('total_size_bytes')}/{stage_summary.get('total_line_count')}")
    print(f"  hold bytes/lines: {hold_summary.get('total_size_bytes')}/{hold_summary.get('total_line_count')}")
    print(f"  foundation stage list: {hygiene.get('foundation_stage_paths_txt')}")
    print(f"  review hold list: {hygiene.get('review_hold_paths_txt')}")
    print(f"  foundation stage status: {hygiene.get('foundation_stage_status_tsv')}")
    print(f"  review hold status: {hygiene.get('review_hold_status_tsv')}")
    dry = hygiene.get("git_add_dry_run") or {}
    print(f"  git-add dry-run index unchanged: {dry.get('cached_unchanged')}")
    diag = hygiene.get("stage_plan_diagnostics") or {}
    print(f"  stage-plan safe: {hygiene.get('stage_plan_safe')}")
    print(f"  stage/hold overlap: {diag.get('stage_hold_overlap_count')}")
    print(f"  dry-run hold overlap: {diag.get('git_add_dry_run_hold_overlap_count')}")
    review = hygiene.get("foundation_cleanup_required_review") or {}
    critical = hygiene.get("foundation_cleanup_critical_gate_review") or {}
    print(f"  foundation cleanup review: {hygiene.get('foundation_cleanup_review_decision')}")
    print(f"  required cleanup paths ok: {review.get('ok_count')}/{review.get('required_path_count')}")
    print(
        "  critical gate paths ok: "
        f"{critical.get('ok_count')}/{critical.get('critical_gate_path_count')} "
        f"missing={len(critical.get('missing_from_repo') or [])} "
        f"dirty-missing-stage={len(critical.get('dirty_missing_from_stage') or [])}"
    )
    print(f"  foundation cleanup stage-ready: {hygiene.get('foundation_cleanup_stage_ready')}")
    print(f"  stage command: {' '.join(hygiene.get('foundation_cleanup_stage_command') or [])}")
    post = hygiene.get("foundation_cleanup_post_stage_verification") or {}
    print(f"  post-stage verification: {post.get('decision')} cached={post.get('cached_count')}")
    print(f"  git-add dry-run output: {hygiene.get('git_add_dry_run_txt')}")
print(f"  next: {report.get('next_allowed_command')}")
print(f"  readiness report: {path}")
adoption_root = Path("/home/andre2/GX1_DATA/reports/entry_foundation_adoption_candidate_20260629_v1")
adoption_candidates = (
    sorted(
        adoption_root.glob("*/ENTRY_FOUNDATION_ADOPTION_CANDIDATE_latest.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    if adoption_root.exists()
    else []
)
if adoption_candidates:
    adoption_path = adoption_candidates[0]
    adoption = json.loads(adoption_path.read_text(encoding="utf-8"))
    print(f"  adoption-candidate: {adoption.get('decision')} ready={adoption.get('candidate_ready_for_activation')}")
    print(f"  adoption activation without vedtak: {adoption.get('activation_allowed_without_vedtak')}")
    print(f"  adoption report: {adoption_path}")

def latest_report(root: Path, filename: str):
    candidates = (
        sorted(
            root.glob(f"*/{filename}"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        if root.exists()
        else []
    )
    if not candidates and (root / filename).exists():
        candidates = [root / filename]
    return candidates[0] if candidates else None

plan_path = latest_report(
    Path("/home/andre2/GX1_DATA/reports/entry_foundation_activation_plan_20260629_v1"),
    "ENTRY_FOUNDATION_ACTIVATION_PLAN_latest.json",
)
if plan_path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    print(f"  activation-plan: {plan.get('decision')} strategy={plan.get('recommended_strategy')}")
    print(f"  activation allowed without vedtak: {plan.get('activation_allowed_without_vedtak')}")
    print(f"  activation-plan report: {plan_path}")

apply_path = latest_report(
    Path("/home/andre2/GX1_DATA/reports/entry_foundation_activation_apply_20260629_v1"),
    "ENTRY_FOUNDATION_ACTIVATION_APPLY_latest.json",
)
if apply_path:
    apply = json.loads(apply_path.read_text(encoding="utf-8"))
    post_commands = apply.get("post_apply_commands") if isinstance(apply.get("post_apply_commands"), list) else []
    print(f"  activation-apply dry-run: {apply.get('decision')} mutation_performed={apply.get('mutation_performed')}")
    print(f"  activation-apply post commands: {len(post_commands)}")
    print(f"  activation-apply report: {apply_path}")

post_apply_path = latest_report(
    Path("/home/andre2/GX1_DATA/reports/entry_foundation_activation_post_apply_20260629_v1"),
    "ENTRY_FOUNDATION_ACTIVATION_POST_APPLY_latest.json",
)
if post_apply_path:
    post_apply = json.loads(post_apply_path.read_text(encoding="utf-8"))
    print(
        "  activation-post-apply: "
        f"{post_apply.get('decision')} "
        f"activation_mutation={post_apply.get('activation_alias_mutation_performed')} "
        f"post_apply_mutations={post_apply.get('post_apply_mutations_performed')}"
    )
    print(f"  activation-post-apply next: {post_apply.get('next_required_action')}")
    print(f"  activation-post-apply report: {post_apply_path}")
PY
  fi
  echo
  echo "Allowed next:"
  echo "  Follow train-readiness next command above."
  echo "  Smoke training stays closed until train-readiness PASS, clean git, and explicit vedtak."
  echo
  echo "Blocked:"
  echo "  generic train/retrain, promote, pin, shadow, paper/live order placement,"
  echo "  candidate train, replay, IQL, promotion review, and legacy live/practice paths."
  echo
  echo "Historical legacy handover is available only with:"
  echo "  GX1_ALLOW_LEGACY_HANDOVER=20260627_ALLOW_LEGACY_HANDOVER bash scripts/gx1_handover.sh"
  echo
  exit 0
fi

bar(){ printf '%s\n' "────────────────────────────────────────────────────────────────────────"; }
# MODE is DERIVED, never asserted (2026-06-11 fix: the hardcoded "BUILD (live stopped)" headline
# contradicted the systemd-active data daemons printed right below it).
_pr_pid=$(cat "$PAPER/paper_runner.pid" 2>/dev/null || true)
if [[ -n "${_pr_pid:-}" ]] && kill -0 "$_pr_pid" 2>/dev/null; then MODE="LIVE (paper_runner alive pid=$_pr_pid)"; else MODE="BUILD (paper_runner stopped; data daemons may still run)"; fi
echo; bar
echo "  GX1 HANDOVER — gjeldende state @ $(date -u '+%Y-%m-%d %H:%M:%SZ')"
echo "  Read first: CLAUDE.md + AGENTS.md + SYSTEM_MAP.md + PROJECT_STATE_artifacts.json (the ONE selection truth)."
echo "  MODE: $MODE. ONE truth = the contract; resolve via gx1_guards.load_decision_artifact."
bar

echo; echo "▌ ACTIVE bundles (one per role — the live chain):"
"$PY" - "$CONTRACT" <<'PY'
import json,sys,os
c=json.load(open(sys.argv[1])); a=c["active"]
def line(role,extra=""):
    v=a[role]; p=v["path"]; ok="OK " if os.path.exists(p) else "MISSING!!"
    print(f"   [{ok}] {role:10s} {p.replace('/home/andre2/GX1_DATA/','')}  {extra}")
line("xgb")
line("v10_entry", "(regime-v4, ctx_cont contract-driven)")
line("v3_exit", "(EXIT_IO_V8)")
line("entry_iql", f"variant={a['entry_iql'].get('active_variant')}  + CONVICTION-GATE overlay")
line("exit_iql", f"variant={a['exit_iql'].get('active_variant')} agg={a['exit_iql'].get('active_aggregator')} K={a['exit_iql'].get('active_k_horizon')}")
op=a["entry_iql"].get("operating_point",{})
print()
print("▌ ENTRY operating point (OPEN-MORE WAVE ARMED 2026-06-13/14 — serve-time overlay, BUNDLE UNCHANGED):")
print(f"   selection={op.get('selection')}  conviction_thr={op.get('conviction_thr')}  sizing={op.get('sizing')}  skip_asia={op.get('skip_asia')}  max_trades={op.get('max_trades')}")
print(f"   live_env: {op.get('live_env')}")
print("   OPEN-MORE: thr -37.71 -> -100 (~+70% takes at full quality); sizing src raw_adv -> margin^2 x inverse-ATR(14).")
print("   TRUE cap-3 mark-to-market account DD = 564 bps, ret/DD 30.5 (inverse-ATR already de-risks the EXTREME-vol tail;")
print("   worst moment is a RECOVERING concurrent-underwater excursion, not lost capital). Nightly op-A/B activates Monday.")
print("   ENTRY-RETRAIN REFUTED (9-agent OOT-verified 2026-06-14): the 'toxic' EXTREME+TREND_UP high-conf LONGs are net-")
print("   PROFITABLE (+5447 bps/77.7%); trough labels do NOT separate them OOT (pre-gate >=0.58 FAILS at 0.501) -> a")
print("   trough-reward refit is futile (oracle -3053 bps). DO NOT retrain entry on this. See memory")
print("   project_gx1_dd_analysis_retrain_refuted_20260614. LAM50 Q-net IS the live entry — do NOT remove the bundle.")
print("   REVERSIBLE: launcher GX1_CONVICTION_THR=-37.71, GX1_SIZING_CONV_SRC=raw_adv (or MARGIN_POW=1.0 for margin^1).")
PY

echo; echo "▌ LIVE stack (status below is live-checked). Relaunch with: bash scripts/launch_live_practice.sh"
if command -v systemctl >/dev/null 2>&1; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  for u in gx1-collector gx1-canonical-incremental; do
    printf "   %-26s %s\n" "$u" "$(systemctl --user is-active "$u" 2>/dev/null || true)"
  done
fi
for pf in paper_runner counterfactual_daemon; do
  p=$(cat "$PAPER/$pf.pid" 2>/dev/null || true)
  if [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null; then st="ALIVE pid=$p"; else st="not running"; fi
  printf "   %-26s %s\n" "$pf" "$st"
done

echo; echo "▌ EDGE BUILDOUT — STEP-0 DONE (5 components, ALL env-gated DEFAULT-OFF → live byte-identical):"
echo "   GX1_SMC_SWEEP_RECLAIM  sweep-then-reclaim (smc_v1)        69cd6441"
echo "   GX1_ROUND_NUMBER       round-number \$-level proximity     542a85df"
echo "   GX1_FVG_FEATURES       FVG M5/M15/H1 proximity            18305072"
echo "   R_WAIT_OPP_K96_SESSCOND session-conditional entry reward  e8173344  (label-only)"
echo "   GX1_SIZING_MODE        conviction/vol position sizing     d775a008"
echo "   STEP-1 overlays wired (17f12208): GX1_SMC_RECLAIM_GATE (falling-knife skip) + GX1_ROUND_NUMBER_WALL,"
echo "     default-OFF. A/B feature-set → \$GX1_DATA/runs/FASE2B_CLEAN_20260608/forward_outcome_step1feats_clean"
echo "     (CLEAN-wave re-augment, uid-aligned with the gate chain; the entry-wave step1feats was parked — its"
echo "     FVG cols predate the M15/H1 look-ahead fix)."
echo "   Verdict: chain is SELECTION-limited; reward-shaping + sizing > features (project_gx1_feature_edge_verdict_20260611)."

echo; echo "▌ DATA: April-2026 is x10-REPAIRED → NO exclusion anywhere (code+memory purged). Use ALL data."

echo; echo "▌ Last 10 commits (newest first):"
git log --oneline -10 | sed 's/^/   /'

echo; echo "▌ Verify (test==serve + rule-9):"
echo "   • rule-9 re-audit (contract-resolves entry-vs-exit wave), from the repo root:"
echo "       $REPO/.venv/bin/python -m gx1.audit.full_state_reaudit --detail"
echo "     (expect Entry 197/197 + Exit 209/209 ALIVE, XGB 0-gain; per-TF EMA alive×5 TF; dip/struct 35/36 alive)."
echo "   • conviction test==serve: live formula reproduces the offline conviction20 decisions (take-rate 0.2000, 1.0000)."

echo; echo "▌ Rollback:"
echo "   • EXIT chain → re-point v3_exit + exit_iql in the contract to the FASE2B_REGIME_V4 history[] entries"
echo "     (v3_exit_fase2b_regime_v4 + exit_iql_train_clean — bundles intact on disk)."
echo "   • ENTRY → unset GX1_CONVICTION_GATE / GX1_SKIP_ASIA (reverts to LAM50 argmax; no artifact restore)."
echo "   • Edge features/overlays → unset their flags (default-OFF). Rollback re-activates BUNDLES, never datasets."

echo; echo "▌ Cleanup state:"
for sup in _SUPERSEDED_20260610 _SUPERSEDED_20260611; do
  if [[ -f "$DATA/runs/$sup/MANIFEST.json" || -f "$DATA/$sup/MANIFEST.json" ]]; then
    mf="$DATA/$sup/MANIFEST.json"; [[ -f "$DATA/runs/$sup/MANIFEST.json" ]] && mf="$DATA/runs/$sup/MANIFEST.json"
    n=$("$PY" -c "import json;print(json.load(open('$mf'))['n_items'])" 2>/dev/null || echo "?")
    sz=$(du -sh "$(dirname "$mf")" 2>/dev/null | cut -f1)
    echo "   • $n items ($sz) reversibly parked → $sup/ (rule-8; MANIFEST.json = restore map). 0 hard-deleted."
  fi
done
echo "   • Memory pruned 2026-06-11: 44 superseded (pre-fase2b) files removed; 34 current kept."

echo; echo "▌ 2026-06-11 FIX-WAVE (audit → fixed + verified, commits e44fd7dc/3585c5c2/…):"
echo "   • CONVICTION-GATE was a LIVE NO-OP (ensemble-only branch) → hoisted to the single-bundle path;"
echo "     flags-OFF parity 200/200 + gate-ON formula 200/200 through the REAL predict(). Re-verify take-rate"
echo "     (~0.20 non-ASIA) IN-VIVO after relaunch."
echo "   • SESSCOND reward fixed (was KeyError-dead + crashed default builds) — now OPT-IN via --variants."
echo "   • FVG M15/H1 forming-bar look-ahead fixed (completed-bars-only; build==serve parity-tested)."
echo "   • Exit-IO contracts PINNED vs GX1_SMC_SWEEP_RECLAIM mutation; sizing fail-closed on missing ATR."
echo "   • ROUND-WALL A/B (12k cement replay, live-parity-checked): NO edge — vetoes 0.30%, removes"
echo "     86%-win trades (−420 bps total), Δbps/take +0.03, cap-3 DD unchanged (−172.3). NOT flipped."

echo; echo "▌ 2026-06-11 LATE WAVE (post-audit):"
echo "   • FALLING-KNIFE A/B DONE (no canonical rebuild needed — one-truth recompute, SMC-parity 0.00000):"
echo "     NO EDGE — vetoes 9.24% of takes that WIN 93.9% (+24.6 bps avg). Veto overlays are now PROVEN"
echo "     arithmetically capped (+2.3% oracle ceiling, losers are only 2.2% of gross). RETIRED as a lever."
echo "   • STEP-2 SESSCOND v1: BOTH variants FAIL the hardened gate vs conviction20 — non-SYM Δbps −3.9 +"
echo "     2026-win 0.8497; SYM Δbps −6.4, Δwin −1.4pp, DD −395. Bundle entry_iql_sesscond_20260611 ="
echo "     REJECTED evidence (kept). NB: SESSCOND argmax opens 25.2% naturally (over-skip cured by reward);"
echo "     SYM had the BEST 2026 bps (30.8). v2 = recalibrated λ (US-strict) + hinge margin + bad_path-fix."
echo "   • BASE34 FREEZE repaired (37 ctx cols frozen since 05-25 — backfilled, daemon fixed+restarted) and"
echo "     rule-9 LIVE-TAIL check wired (launcher hard-fail preflight + hourly daemon self-check)."
echo "   • Edge headroom QUANTIFIED: open-more (~+45% extrap; band [-67,-34.2) replay = the missing experiment)"
echo "     > sizing (+14% at 2x top-decile) >> exit/veto (exhausted). LAM50 reward depression documented."

echo; echo "▌ What's next:"
echo "   1. WATCH LIVE (Monday open): open-more -100 + margin² + skip-ASIA is armed; the nightly op-A/B leg"
echo "      (gx1.scripts.nightly_op_comparison) grades it vs conviction67 on resolved trades. Watch live DD/win."
echo "   2. ENTRY: do NOT retrain on the toxic-cluster (OOT-refuted — trough labels unlearnable, family is profitable)."
echo "      The productive entry direction is OPEN-MORE (selection breadth) + the nightly continuous-learning loop,"
echo "      NOT suppressing a profitable regime. Marginal DD lever (if wanted): GX1_SIZING_ATR_REF_BPS 14->10 (DD 564->537,"
echo "      costs ~2000 bps realized; reversible env-flag, no retrain)."
echo "   3. Reward-fix bundle v2 at next entry-IQL refit (own vedtak): _SYM family + bad_path positive-part gate +"
echo "      truncation mask + recalibrated SESSCOND λ. STEP-3 round+FVG ctx cascade (leak-free cols ready)"
echo "      → STEP-4 sweep SEQ → FEAT-6 M1-native FVG for the exit. Own-vedtak each."
bar; echo
