#!/usr/bin/env bash
# GX1 HANDOVER — print the CURRENT gjeldende state for a fresh session.
#
# Re-runnable orientation: reads PROJECT_STATE_artifacts.json (the ONE selection truth) live,
# checks the live stack, and prints the operating point + verify/rollback/next steps.
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
echo; bar
echo "  GX1 HANDOVER — gjeldende state @ $(date -u '+%Y-%m-%d %H:%M:%SZ')"
echo "  ONE selection truth = PROJECT_STATE_artifacts.json (resolve via gx1_guards.load_decision_artifact)"
bar

echo; echo "▌ ACTIVE bundles (one per role — running LIVE now):"
"$PY" - "$CONTRACT" <<'PY'
import json,sys,os
c=json.load(open(sys.argv[1]))
a=c["active"]
def line(role,extra=""):
    v=a[role]; p=v["path"]; ok="OK " if os.path.exists(p) else "MISSING!!"
    print(f"   [{ok}] {role:10s} {p.replace('/home/andre2/GX1_DATA/','')}  {extra}")
line("xgb")
line("v10_entry")
line("v3_exit", f"({a['v3_exit'].get('note','')[:0]}EXIT_IO_V8)")
line("entry_iql", f"variant={a['entry_iql'].get('active_variant')}")
line("exit_iql", f"variant={a['exit_iql'].get('active_variant')} agg={a['exit_iql'].get('active_aggregator')} K={a['exit_iql'].get('active_k_horizon')}")
op=a["entry_iql"].get("operating_point",{})
print()
print("▌ ENTRY operating point (CEMENTED 2026-06-10, serve-time overlay — entry_iql BUNDLE UNCHANGED):")
print(f"   selection      : {op.get('selection')}")
print(f"   conviction_thr : {op.get('conviction_thr')}   (open top-20% by raw_adv = best-TAKE-Q − SKIP-Q, side=argmax TAKE-Q)")
print(f"   skip_asia      : {op.get('skip_asia')}        (block ASIA-session entries, per-year win floor)")
print(f"   max_trades     : {op.get('max_trades')}            (DD-validated portfolio cap)")
print(f"   live_env       : {op.get('live_env')}")
print("   LAM50 NOTE     : the live entry IS the LAM50-REWARD Q-net (entry_iql_clean). Only the OLD")
print("                    LAM50/VOLUME-FIRST *operating point* (argmax-SKIP) is superseded. DO NOT remove the bundle.")
PY

echo; echo "▌ LIVE stack status:"
if command -v systemctl >/dev/null 2>&1; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  for u in gx1-collector gx1-canonical-incremental; do
    printf "   %-26s %s\n" "$u" "$(systemctl --user is-active "$u" 2>/dev/null || echo unknown)"
  done
fi
for pf in paper_runner counterfactual_daemon; do
  p=$(cat "$PAPER/$pf.pid" 2>/dev/null || true)
  if [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null; then st="ALIVE pid=$p"; else st="not running"; fi
  printf "   %-26s %s\n" "$pf" "$st"
done
echo "   (re-run: bash scripts/launch_live_practice.sh  — idempotent, pins conviction-gate+skip-ASIA)"

echo; echo "▌ Key commits (entry-selection cement + cleanup):"
git log --oneline -6 | sed 's/^/   /'

echo; echo "▌ Verify (test==serve + rule-9 liveness):"
echo "   • test==serve : live conviction formula reproduces the offline conviction20 decisions (take-rate 0.2000, match 1.0000)."
echo "   • rule-9      : python gx1/audit/full_state_reaudit.py  (defaults -> FASE2B_CLEAN_20260608; expect Entry 196/197 + Exit 208/209 ALIVE)."
echo "   • feature-liveness pre-cement: python -m gx1.audit.feature_liveness --strict --v10-bundle <active> --xgb-bundle <active>"

echo; echo "▌ Rollback:"
echo "   • EXIT  : re-point v3_exit + exit_iql in PROJECT_STATE_artifacts.json to the FASE2B_REGIME_V4 history"
echo "             entries (v3_exit_fase2b_regime_v4 + exit_iql_train_clean — both bundles intact on disk)."
echo "   • ENTRY : unset GX1_CONVICTION_GATE / GX1_SKIP_ASIA (reverts to LAM50 argmax; no artifact restore)."
echo "   • Rollback re-activates BUNDLES, never datasets."

echo; echo "▌ Cleanup state:"
if [[ -f "$DATA/_SUPERSEDED_20260610/MANIFEST.json" ]]; then
  n=$("$PY" -c "import json;print(json.load(open('$DATA/_SUPERSEDED_20260610/MANIFEST.json'))['n_items'])" 2>/dev/null || echo "?")
  sz=$(du -sh "$DATA/_SUPERSEDED_20260610" 2>/dev/null | cut -f1)
  echo "   • $n superseded items ($sz) reversibly parked -> _SUPERSEDED_20260610/ (rule-8; MANIFEST.json = restore map)."
  echo "   • 0 hard-deleted. Hard-delete is a SEPARATE rule-5 step (backup->inventory->dry-run->user confirm)."
fi

echo; echo "▌ What's next:"
echo "   • IN-VIVO: confirm live take-rate ~20% non-ASIA (vs LAM50 ~8%), zero ASIA takes (accrues over EU/US sessions)."
echo "   • FEATURE WAVE #2 (own vedtak per retrain): ema20×50 + OOT-tail+rule-9-gated selection features"
echo "     (FVG-proximity, sweep-then-reclaim, round-number levels, vol-regime-transition, session-cond reward,"
echo "     RV-term-slope, conviction×regime); cross-asset (real-yields>DXY>GSR>VIX, level/residual not momentum)."
echo "   • Bake skip-ASIA as a LEARNED session-conditional entry reward at the next entry-IQL retrain."
echo "   • Deferred build-path repoints before parking the 7 KEEP_UNSAFE: GX1_B9_DECISIONS_PATH +"
echo "     posthoc_session_strategyf_eval CEMENT/FASE2B baselines."
bar; echo
