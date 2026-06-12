#!/usr/bin/env bash
set -euo pipefail
# scripts/gx1_candidate_gate.sh — CANDIDATE Entry-IQL AUTO-GATE harness (ladder wave 2026-06-12).
#
# One command: takes a PENDING candidate bundle (nightly-refit output under
# GX1_DATA/reports/online_iql/) and emits PASS/FAIL + a verdict json. EVIDENCE ONLY — this
# script NEVER writes PROJECT_STATE_artifacts.json and NEVER moves/deletes bundles; promotion
# stays a MANUAL contract flip (rule 8). The candidate enters the chain ONLY via the explicit
# --entry-iql-lock arg of the established inference entrypoint (explicit CLI args are the
# rule-8-sanctioned override; every DEFAULT below stays contract-resolved + fail-closed).
#
# Usage: gx1_candidate_gate.sh <candidate_bundle_dir> [--quick] [--dry-run]
#                              [--fold FOLD_1] [--out-root DIR]
#   --quick    cap phase6 at --max-candidates 3000 (PENDING-read; full 12k before any flip)
#   --dry-run  sanity + resolution + print the exact commands; runs nothing heavy, writes no
#              verdict under candidate_gates/ (a dry-run must never look like gate evidence)
#
# Chain (mirrors the PROVEN launchers — do not invent a new invocation):
#   (1) sanity     every trained_models_v1/*.pt loads; state_dim == summary feature count ==
#                  the ACTIVE entry bundle's arch (warm-start arch-match); then a REAL
#                  EntryIQLV2Adapter.load() — the load-bearing check.
#   (2) inference  candidate decisions.parquet via v12_phase1_entry_iql_inference (the same
#                  entrypoint that produced the 2026-06-08 cement decisions; pattern =
#                  GX1_DATA/runs/FASE2B_CLEAN_20260608/launch_entry_inference_clean.sh).
#   (3) phase6     v12_phase6_joint_validation --gate against the contract-ACTIVE exit chain
#                  (pattern + env = FASE2B_CLEAN_20260608/launch_phaseD_gate_clean.sh:36,58-67).
#   (4) posthoc    gx1.research.posthoc_session_strategyf_eval (hardened skip-ASIA verdict vs
#                  COSTFIX + live-fase2b baselines; exit 0 iff PASS).
#   stress         NOT runnable for a candidate: v12_counterfactual_replay --mode stress reads
#                  the LIVE journal's recorded v12_decision actions (live policy only). The
#                  candidate behavioral analog (--mode variants --bundle-dir <cand>) is already
#                  run by gx1_nightly_learning.sh (Track-B candidate_shadow). Recorded honestly
#                  as 'live-policy only' in the verdict — never faked.
#
# PASS = sanity OK AND phase6 --gate exit 0 AND posthoc exit 0 (fail-closed: any step that
# cannot run ⇒ FAIL, never a silent skip). Verdict json →
#   GX1_DATA/reports/v12_paper_runs/nightly_learning/candidate_gates/<candidate>_<date>.json

REPO=/home/andre2/src/GX1_ENGINE
PY=$REPO/.venv/bin/python
# cwd-independent: every `python -m gx1.…` below needs the repo on PYTHONPATH —
# without this the script only worked when invoked FROM the repo root
# (discovered when the first background/full run failed with No module named).
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
GATES_DIR=/home/andre2/GX1_DATA/reports/v12_paper_runs/nightly_learning/candidate_gates
TS=$(date -u +%Y%m%dT%H%M%SZ)
TODAY=$(date -u +%Y%m%d)

usage() { grep '^# Usage' -A4 "$0" | sed 's/^# \{0,1\}//'; exit 2; }

CAND="" ; QUICK=false ; DRYRUN=false ; FOLD=FOLD_1 ; OUT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)    QUICK=true ;;
    --dry-run)  DRYRUN=true ;;
    --fold)     FOLD="$2"; shift ;;
    --out-root) OUT_ROOT="$2"; shift ;;
    -h|--help)  usage ;;
    -*)         echo "FATAL: unknown flag $1"; usage ;;
    *)          [[ -n "$CAND" ]] && { echo "FATAL: multiple candidate dirs given"; usage; }; CAND="$1" ;;
  esac
  shift
done
[[ -n "$CAND" ]] || usage

# --- Fail-closed candidate preflight ---------------------------------------------------------
CAND=$(readlink -f "$CAND")
[[ -d "$CAND" ]] || { echo "FATAL: candidate bundle dir missing: $CAND"; exit 1; }
[[ -f "$CAND/summary_v1.json" ]] || { echo "FATAL: no summary_v1.json in $CAND (not an Entry-IQL bundle)"; exit 1; }
ls "$CAND"/trained_models_v1/*.pt >/dev/null 2>&1 \
  || { echo "FATAL: no trained_models_v1/*.pt in $CAND — nothing to gate"; exit 1; }
CAND_NAME=$(basename "$CAND")

[[ -n "$OUT_ROOT" ]] || OUT_ROOT="$GATES_DIR/runs"
GATE_OUT="$OUT_ROOT/${CAND_NAME}_${TS}"
# A dry-run must leave NO debris under the gate-evidence tree (2026-06-12 finding)
$DRYRUN && GATE_OUT=$(mktemp -d /tmp/gx1_gate_dryrun.XXXXXX)
mkdir -p "$GATE_OUT"

# --- (1) SANITY — real loads, fail-closed (also derives the candidate's variant/folds) --------
# state_dim must equal len(summary feature_names_v1) (EntryIQLV2Adapter.load enforces it,
# entry_iql_v2_adapter.py:145) AND the ACTIVE entry bundle's n_features (a nightly warm-start
# candidate that drifted arch can never be live-compatible — catch it here, loud).
SANITY_JSON="$GATE_OUT/sanity.json"
CAND="$CAND" SANITY_JSON="$SANITY_JSON" GATE_FOLD="$FOLD" PYTHONPATH=$REPO $PY - <<'PYEOF'
import json, os, sys
from pathlib import Path
import torch
from gx1_guards.artifacts import load_decision_artifact   # rule 8: contract-only resolution
from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter

cand = Path(os.environ["CAND"]); fold = os.environ["GATE_FOLD"]
summary = json.loads((cand / "summary_v1.json").read_text())
feat = summary["feature_names_v1"]
active = Path(load_decision_artifact("entry_iql"))
active_nfeat = json.loads((active / "summary_v1.json").read_text())["n_features_v1"]

ckpts, variants = {}, set()
for fp in sorted((cand / "trained_models_v1").glob("*.pt")):
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    sd, var, fid = int(ck["state_dim"]), str(ck["variant"]), str(ck["fold_id"])
    assert sd == len(feat), f"{fp.name}: state_dim {sd} != summary feature count {len(feat)}"
    assert sd == int(active_nfeat), f"{fp.name}: state_dim {sd} != ACTIVE entry arch {active_nfeat}"
    for key in ("feature_means", "feature_stds"):
        vals = torch.as_tensor(ck[key], dtype=torch.float64)
        assert len(vals) == sd and bool(torch.isfinite(vals).all()), f"{fp.name}: bad {key}"
    variants.add(var); ckpts[fp.name] = {"variant": var, "fold": fid, "state_dim": sd}
assert len(variants) == 1, f"ambiguous candidate: multiple variants {sorted(variants)} — refusing to guess"
variant = variants.pop()
assert (cand / "trained_models_v1" / f"{variant}_{fold}.pt").is_file(), \
    f"gate fold {fold} missing for variant {variant}"

# The load-bearing check: the exact adapter load the inference step will perform.
adapter = EntryIQLV2Adapter.load(artifact_root=cand, variant=variant, fold_id=fold,
                                 aggregator="mean", beta=1.0, prefer_cuda=False,
                                 min_advantage_bps=0.0)
out = {"candidate": str(cand), "variant": variant, "gate_fold": fold,
       "n_features": len(adapter.feature_names), "checkpoints": ckpts,
       "active_entry_bundle": str(active), "active_entry_n_features": int(active_nfeat),
       "ok": True}
Path(os.environ["SANITY_JSON"]).write_text(json.dumps(out, indent=2))
print(f"SANITY OK: variant={variant} folds={[c['fold'] for c in ckpts.values()]} "
      f"state_dim={len(feat)} (== ACTIVE arch) adapter-load OK")
PYEOF
VARIANT=$($PY -c "import json;print(json.load(open('$SANITY_JSON'))['variant'])")

# --- Resolve the ACTIVE exit chain through the contract (rule 8) ------------------------------
# Dataset paths derive from the ACTIVE exit_iql bundle's WAVE dir — the full_state_reaudit
# 2026-06-11 lesson: never hardcode one wave for both arms; the exit bundle, its scored per-bar
# build and its canonical MUST come from the SAME wave (train==serve: launch_phaseD_gate_clean.sh
# pairs exit_iql_retrain_clean_20260609 with that wave's exit_per_bar_scored_clean + canonical).
eval "$(PYTHONPATH=$REPO $PY - <<'PYEOF'
from pathlib import Path
from gx1_guards.artifacts import load_decision_entry
e = load_decision_entry("exit_iql")
bundle = Path(e["path"]); wave = bundle.parent
print(f"EXIT_BUNDLE={bundle}")
print(f"EXIT_VARIANT={e['active_variant']}")
print(f"WAVE_DIR={wave}")
PYEOF
)"
SCORED="$WAVE_DIR/exit_per_bar_scored_clean"
CANON="$WAVE_DIR/CANONICAL_FEATURES_V3_PLUS5/canonical_features_v3_plus5.parquet"
FWD="$WAVE_DIR/forward_outcome_clean/per_week"
{ [ -d "$SCORED/per_week" ] && [ -n "$(find "$SCORED/per_week" -name '*.parquet' -print -quit)" ]; } \
  || { echo "FATAL: scored per-bar missing/empty: $SCORED/per_week (wave layout changed? fix paths here deliberately)"; exit 1; }
[ -f "$CANON" ] || { echo "FATAL: canonical features missing: $CANON (must match what the ACTIVE exit_iql built against)"; exit 1; }
{ [ -d "$FWD" ] && [ -n "$(find "$FWD" -name '*.parquet' -print -quit)" ]; } \
  || { echo "FATAL: forward_outcome per_week missing/empty: $FWD (uid-aligned source for the scored per-bar)"; exit 1; }

# --- RAM headroom (AGENTS.md hard ceiling — the 2026-06-10 OOM crashed the PC) ----------------
# Full gate loads ~5.7M per-bar rows (~15-20 GB); --quick (3000 cands) ~1/4 of that.
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
NEED_GB=16; $QUICK && NEED_GB=10
if ! $DRYRUN && [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
  echo "FATAL: only ${AVAIL_GB}GB RAM available (< ${NEED_GB}GB) — refusing to start the gate (blocked_low_ram)"
  exit 1
fi

# train==serve env parity — EXACTLY launch_phaseD_gate_clean.sh:36. Without GX1_EXIT_AUGMENT_64=1
# the gate loader's _compute_exit_aug64 returns None → 64 exit features missing →
# FEATURE_COVERAGE_FATAL (the 2026-06-10 lesson). REGIME flags for ctx parity.
export GX1_EXIT_AUGMENT_64=1 GX1_REGIME_V4=1 GX1_TREND_REGIME_FROM_D1=1

PH6_EXTRA=()
$QUICK && PH6_EXTRA+=(--max-candidates 3000)

# 2026-06-12: module PROMOTED out of _legacy_disabled (NO-OLD-CODE — it is
# load-bearing for the nightly gate); all its hardcoded defaults were removed.
CMD_INFER=("$PY" -m gx1.scripts.v12_phase1_entry_iql_inference
  --entry-iql-lock "$CAND"
  --forward-outcome-dir "$FWD"
  --out-root "$GATE_OUT/entry_decisions"
  --variant "$VARIANT" --fold-id "$FOLD")
# CANDIDATE OVERRIDE — gate evaluation only, never decisioning: the candidate bundle is passed
# as an EXPLICIT --entry-iql-lock; nothing here changes how live/serve resolves entry_iql
# (that stays gx1_guards.load_decision_artifact on PROJECT_STATE_artifacts.json — rule 8).
CMD_PH6=("$PY" -m gx1.scripts.v12.v12_phase6_joint_validation
  --v3tracked-lock "$SCORED"
  --exit-iql-v5-lock "$EXIT_BUNDLE"
  --entry-iql-decisions "__DECISIONS__"
  --variant "$EXIT_VARIANT" --fold-id FOLD_1 --skip-v12-on
  --canonical-features "$CANON"
  --out-root "$GATE_OUT/phase6" --gate "${PH6_EXTRA[@]}")
# DECISIVE baseline = the ACTIVE volbal chain when its per-candidate CSV exists
# (generated once via this same harness with --entry-iql-lock <ACTIVE bundle>);
# the tool's built-in default is the SUPERSEDED lam50 chain — comparing against
# it inflates PASS (volbal beat lam50 on all axes; 2026-06-12 finding).
VOLBAL_BASELINE="$GATES_DIR/baselines/volbal_per_candidate_V12_OFF.csv"
CMD_POSTHOC=("$PY" -m gx1.research.posthoc_session_strategyf_eval "$GATE_OUT/phase6/per_candidate_V12_OFF.csv")
if $QUICK; then
  # quick mode caps candidates at 3000 — a FULL 12k baseline would fail the
  # posthoc coverage>=80% check on N alone. Quick nights = phase6 floors +
  # per-year posthoc floor only; the DECISIVE volbal comparison runs in FULL
  # mode before any flip.
  CMD_POSTHOC+=(--fase2b-baseline "")
  BASELINE_NOTE="quick mode: decisive volbal comparison SKIPPED (coverage-incomparable) — run FULL before any flip"
elif [[ -f "$VOLBAL_BASELINE" ]]; then
  CMD_POSTHOC+=(--fase2b-baseline "$VOLBAL_BASELINE")
  BASELINE_NOTE="ACTIVE volbal chain ($VOLBAL_BASELINE)"
else
  BASELINE_NOTE="DEFAULT lam50 chain (SUPERSEDED by volbal — PASS may be inflated; generate $VOLBAL_BASELINE)"
fi

echo "=== CANDIDATE-GATE MANIFEST ($(date -u)) ==="
echo "  commit:       $(git -C "$REPO" rev-parse HEAD)"
echo "  candidate:    $CAND  (variant=$VARIANT fold=$FOLD)"
echo "  ckpt sha256:  $(sha256sum "$CAND"/trained_models_v1/*.pt | awk '{print substr($1,1,12), $2}' | xargs -I{} echo -n '{} ')"
echo "  exit bundle:  $EXIT_BUNDLE (ACTIVE via contract, variant=$EXIT_VARIANT)"
echo "  scored:       $SCORED"
echo "  fwd:          $FWD"
echo "  canon:        $CANON"
echo "  env:          GX1_EXIT_AUGMENT_64=1 GX1_REGIME_V4=1 GX1_TREND_REGIME_FROM_D1=1 GX1_REPLAY_WORKERS=${GX1_REPLAY_WORKERS:-<tool default: cores-4 capped 12>}"
echo "  mode:         quick=$QUICK dry_run=$DRYRUN  ram_avail=${AVAIL_GB}GB"
echo "  out:          $GATE_OUT"
echo "  posthoc base: $BASELINE_NOTE"
echo "  WAVE CAVEAT:  candidate entry decisions are inferred on the EXIT wave's"
echo "                forward_outcome ($(basename "$WAVE_DIR")) — the same pairing as the cement"
echo "                evidence, but NOT the candidate's training wave (FASE2B_REGIME_V4)."
echo "                Entry-wave-keyed gate inputs are a tracked follow-up; read PASS/FAIL"
echo "                with this in mind (2026-06-11 wave-mismatch lesson)."
echo "  NOTE: contract is READ-ONLY here; PASS != promotion (manual flip, rule 8)."
echo "=============================================="

if $DRYRUN; then
  echo "[dry-run] step 2 (inference): ${CMD_INFER[*]}"
  echo "[dry-run] step 3 (phase6):    ${CMD_PH6[*]}"
  echo "[dry-run] step 4 (posthoc):   ${CMD_POSTHOC[*]}"
  echo "[dry-run] verdict would go →  $GATES_DIR/${CAND_NAME}_${TS}_<mode>.json"
  echo "[dry-run] DONE (sanity PASSED; nothing executed, no verdict written)"
  exit 0
fi

# Rule 2: never start a Phase-6 run on a dirty tree (sanity/dry-run above are reads, not runs).
if [ -n "$(git -C "$REPO" status --short)" ]; then
  echo "FATAL: git tree dirty — rule 2. Commit/stash first:"; git -C "$REPO" status --short; exit 1
fi

# --- (2) candidate entry-decisions inference --------------------------------------------------
echo ">>> [2/4] candidate entry-IQL inference → decisions.parquet"
"${CMD_INFER[@]}" 2>&1 | tee "$GATE_OUT/entry_inference.log"
# Newest inside OUR OWN just-created out-root (not artifact selection — rule 8 untouched).
DECISIONS=$(find "$GATE_OUT/entry_decisions" -name decisions.parquet | sort | tail -1)
[ -f "$DECISIONS" ] || { echo "FATAL: inference produced no decisions.parquet"; exit 1; }
N_TAKE=$($PY -c "
import pandas as pd
d = pd.read_parquet('$DECISIONS', columns=['action_label_v1'])
print(int(d['action_label_v1'].isin(['TAKE_LONG_NOW','TAKE_SHORT_NOW']).sum()))")
echo "  decisions: $DECISIONS  (TAKE rows: $N_TAKE)"
# < 50 takes can't support any gate statistic (mirrors the phase6 per-side Spearman n>=50 floor).
[ "$N_TAKE" -ge 50 ] || { echo "FATAL: candidate takes only $N_TAKE (<50) — degenerate policy, gate FAIL"; exit 1; }

# --- (3) phase6 joint-validation --gate (PASS/FAIL, fail-closed in the tool) ------------------
echo ">>> [3/4] phase6 joint-validation --gate (V12_OFF) vs ACTIVE exit chain"
PH6_RC=0
CMD_PH6=("${CMD_PH6[@]/__DECISIONS__/$DECISIONS}")
"${CMD_PH6[@]}" 2>&1 | tee "$GATE_OUT/phase6.log" || PH6_RC=$?

# --- (4) posthoc hardened skip-ASIA verdict (exit 0 iff PASS) ---------------------------------
echo ">>> [4/4] posthoc session/strategy-F eval vs cement baselines"
POSTHOC_RC=0
if [ -f "$GATE_OUT/phase6/per_candidate_V12_OFF.csv" ]; then
  "${CMD_POSTHOC[@]}" 2>&1 | tee "$GATE_OUT/posthoc.log" || POSTHOC_RC=$?
else
  echo "  phase6 produced no per-candidate CSV — posthoc impossible (fail-closed)"
  POSTHOC_RC=1
fi

# --- Verdict json (the ONLY write outside this run's out dir) ---------------------------------
# Filename carries TS + mode (2026-06-12 finding: a (candidate, day) key let a
# quick run and a later full run silently overwrite each other's evidence).
mkdir -p "$GATES_DIR"
MODE_TAG=$($QUICK && echo quick || echo full)
VERDICT="$GATES_DIR/${CAND_NAME}_${TS}_${MODE_TAG}.json"
GATE_OUT="$GATE_OUT" SANITY_JSON="$SANITY_JSON" CAND="$CAND" VARIANT="$VARIANT" \
PH6_RC="$PH6_RC" POSTHOC_RC="$POSTHOC_RC" N_TAKE="$N_TAKE" QUICK="$QUICK" \
DECISIONS="$DECISIONS" EXIT_BUNDLE="$EXIT_BUNDLE" VERDICT="$VERDICT" \
BASELINE_NOTE="$BASELINE_NOTE" WAVE_DIR="$WAVE_DIR" $PY - <<'PYEOF'
import json, os
from datetime import datetime, timezone
from pathlib import Path
out_dir = Path(os.environ["GATE_OUT"])
ph6_rc, post_rc = int(os.environ["PH6_RC"]), int(os.environ["POSTHOC_RC"])
gate_json = out_dir / "phase6" / "phase6_gate_v1.json"
gate = json.loads(gate_json.read_text()) if gate_json.is_file() else None
passed = (ph6_rc == 0) and (post_rc == 0)   # sanity already hard-failed the script if broken
verdict = {
    "candidate": os.environ["CAND"],
    "variant": os.environ["VARIANT"],
    "built_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": "quick" if os.environ["QUICK"] == "true" else "full",
    "gate_out_dir": str(out_dir),
    "steps": {
        "sanity": json.loads(Path(os.environ["SANITY_JSON"]).read_text()),
        "entry_inference": {"decisions": os.environ["DECISIONS"], "n_take": int(os.environ["N_TAKE"])},
        "phase6_gate": {"rc": ph6_rc, "passed": ph6_rc == 0,
                        "exit_bundle": os.environ["EXIT_BUNDLE"],
                        "checks": (gate or {}).get("checks"),
                        "per_year": (gate or {}).get("per_year")},
        "posthoc": {"rc": post_rc, "passed": post_rc == 0, "log": str(out_dir / "posthoc.log")},
        # Honest: journal-stress reads LIVE decisions — a candidate has no journal. Its
        # behavioral shadow = nightly Track-B (counterfactual --mode variants --bundle-dir).
        "stress": {"ran": False, "note": "live-policy only — journal-based stress cannot "
                   "evaluate a candidate; see nightly candidate_shadow (Track-B) instead"},
    },
    "passed": bool(passed),
    "posthoc_baseline": os.environ["BASELINE_NOTE"],
    # HONESTY (2026-06-12 adversarial finding, 2026-06-11 wave-mismatch class):
    # entry decisions are inferred on the EXIT wave's forward_outcome — the same
    # pairing as the cement evidence but NOT the candidate's training wave.
    "wave_caveat": (f"entry inputs = EXIT wave {os.environ['WAVE_DIR']} forward_outcome; "
                    f"candidate trained on FASE2B_REGIME_V4 — entry-wave-keyed gate "
                    f"inputs are a tracked follow-up"),
    "promotion": "MANUAL contract flip only (rule 8) — this verdict is evidence, not promotion",
}
Path(os.environ["VERDICT"]).write_text(json.dumps(verdict, indent=2, default=str))
print(f"\n{'='*70}\nCANDIDATE GATE: {'PASS' if passed else 'FAIL'}  (phase6 rc={ph6_rc}, posthoc rc={post_rc})")
print(f"verdict → {os.environ['VERDICT']}\n{'='*70}")
raise SystemExit(0 if passed else 1)
PYEOF
