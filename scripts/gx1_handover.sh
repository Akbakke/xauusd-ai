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
line("v10_entry", "(regime-v4, ctx_cont 123)")
line("v3_exit", "(EXIT_IO_V8)")
line("entry_iql", f"variant={a['entry_iql'].get('active_variant')}  + CONVICTION-GATE overlay")
line("exit_iql", f"variant={a['exit_iql'].get('active_variant')} agg={a['exit_iql'].get('active_aggregator')} K={a['exit_iql'].get('active_k_horizon')}")
op=a["entry_iql"].get("operating_point",{})
print()
print("▌ ENTRY operating point (CEMENTED 2026-06-10 — serve-time overlay, entry_iql BUNDLE UNCHANGED):")
print(f"   selection={op.get('selection')}  conviction_thr={op.get('conviction_thr')}  skip_asia={op.get('skip_asia')}  max_trades={op.get('max_trades')}")
print(f"   live_env: {op.get('live_env')}")
print("   LAM50 NOTE: the live entry IS the LAM50-REWARD Q-net served via the conviction-gate; only the OLD")
print("               LAM50/VOLUME-FIRST argmax operating-point is superseded. Do NOT remove the bundle.")
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
if [[ -f "$DATA/_SUPERSEDED_20260610/MANIFEST.json" ]]; then
  n=$("$PY" -c "import json;print(json.load(open('$DATA/_SUPERSEDED_20260610/MANIFEST.json'))['n_items'])" 2>/dev/null || echo "?")
  sz=$(du -sh "$DATA/_SUPERSEDED_20260610" 2>/dev/null | cut -f1)
  echo "   • $n items ($sz) reversibly parked → _SUPERSEDED_20260610/ (rule-8; MANIFEST.json = restore map). 0 hard-deleted."
fi
echo "   • Memory pruned 2026-06-11: 44 superseded (pre-fase2b) files removed; 34 current kept."

echo; echo "▌ What's next:"
echo "   1. STEP-1 entry-overlay A/B over forward_outcome_step1feats (falling-knife skip + round-number wall),"
echo "      OOT-tail + rule-9 gated (ALL data incl. repaired April, cap-3 DD floor). Flip the winners live."
echo "   2. STEP-2 reward refit (R_WAIT_OPP_K96_SESSCOND_SYM, IQL-frozen) → STEP-3 round+FVG ctx_cont cascade →"
echo "      STEP-4 sweep SEQ cascade. Each is an own-vedtak gated retrain."
echo "   3. FEAT-6: M1-NATIVE FVG (+sweep) for the M1 exit (compute on the M1 tape → exit_io → V3). Exit is ALWAYS M1."
bar; echo
