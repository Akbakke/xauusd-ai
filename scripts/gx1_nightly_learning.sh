#!/usr/bin/env bash
# GX1 nightly continuous-learning loop — ladder wave (user vedtak «Ja til alle 3» 2026-06-12).
#
# ANALYSIS legs run every night (read-only + reports, no vedtak needed):
#   1. RAM guard (abort <8 GB available — AGENTS.md OOM hard ceiling)
#   2. Accumulate per-trade verdicts (trade_verdicts_*.jsonl → regret_dataset.parquet)
#   3. Build/refresh the live-regret replay buffer for MATURED days (D-8..D-2; 25h K-aging)
#   4. KS distribution-drift check (rule-9 drift leg — ADVISORY, never blocks)
#   5. Nightly report JSON in nightly_learning/
#
# REFIT leg (CLAUDE.md rule 3 — NEVER runs without a standing vedtak):
#   Armed ONLY when $STANDING_VEDTAK_FILE exists and contains a vedtak id.
#   Then: warm-start IQL refit on the rolling buffer → candidate bundle under
#   online_iql/ → Track-B style shadow-score of the candidate over recent journals.
#   The candidate is PENDING — this script NEVER flips PROJECT_STATE_artifacts.json
#   (rule 8: promote is a manual contract flip after gates).
#
# Scheduling: systemd --user gx1-nightly-learning.timer (03:30 UTC).
# Manual run: bash scripts/gx1_nightly_learning.sh
set -uo pipefail

REPO=/home/andre2/src/GX1_ENGINE
PY=$REPO/.venv/bin/python
PAPER_DIR=/home/andre2/GX1_DATA/reports/v12_paper_runs
CF_DIR=$PAPER_DIR/counterfactual_reports
OUT_DIR=$PAPER_DIR/nightly_learning
REPLAY_DIR=/home/andre2/GX1_DATA/reports/online_replay
ONLINE_IQL_DIR=/home/andre2/GX1_DATA/reports/online_iql
STANDING_VEDTAK_FILE=/home/andre2/GX1_DATA/config/nightly_refit_standing_vedtak.txt
SUFFIX="${GX1_NIGHTLY_SUFFIX:-conviction67sized_skipasia_pure_phase6}"
VARIANT="${GX1_NIGHTLY_VARIANT:-R_WAIT_OPP_K96_LAM50_SYM}"
TODAY=$(date -u +%Y%m%d)
mkdir -p "$OUT_DIR" "$REPLAY_DIR"

REPORT="$OUT_DIR/nightly_report_${TODAY}.json"
declare -A STATUS

log() { echo "[$(date -u +%H:%M:%SZ)] [nightly] $*"; }

# ── 1. RAM guard (AGENTS.md hard ceiling) ────────────────────────────────────
AVAIL_GB=$(free -g | awk 'NR==2{print $7}')
if (( AVAIL_GB < 8 )); then
    log "ABORT: only ${AVAIL_GB} GB RAM available (<8) — a heavy build is resident; not stacking on top."
    exit 1
fi
log "RAM ok (${AVAIL_GB} GB available)"

# rule 2 note: analysis legs are read-only; the REFIT leg hard-requires a clean tree.
GIT_DIRTY=$(git -C "$REPO" status --short | head -1)
[[ -n "$GIT_DIRTY" ]] && log "WARNING: repo tree is dirty — refit leg will refuse to run."

# ── 2. Verdict accumulation → regret dataset ─────────────────────────────────
log "accumulating per-trade verdicts → regret_dataset.parquet"
if PYTHONPATH=$REPO $PY - <<'EOF'
import glob, json, sys
import pandas as pd
rows = []
for fp in sorted(glob.glob("/home/andre2/GX1_DATA/reports/v12_paper_runs/counterfactual_reports/trade_verdicts_*.jsonl")):
    for line in open(fp):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
if not rows:
    print("[nightly] no verdicts yet — skipping regret dataset"); sys.exit(0)
df = pd.DataFrame(rows).drop_duplicates(subset=["trade_id"], keep="last")
out = "/home/andre2/GX1_DATA/reports/v12_paper_runs/nightly_learning/regret_dataset.parquet"
df.to_parquet(out, index=False)
r = df[df.resolved == True]  # noqa: E712
print(f"[nightly] regret dataset: {len(df)} trades ({len(r)} resolved) → {out}")
if len(r):
    print(f"[nightly]   false_take={int(r.false_take.sum())} wrong_side={int(r.wrong_side.sum())} "
          f"held_too_short={int(r.held_too_short.sum())} good={int(r.good_take.sum())} "
          f"hold_regret_total={r.hold_regret_bps.fillna(0).sum():.0f}bps")
EOF
then STATUS[verdicts]=ok; else STATUS[verdicts]=FAIL; fi

# ── 3. Rolling live-regret replay buffer (matured days only) ─────────────────
FROM=$(date -u -d "8 days ago" +%Y%m%d)
TO=$(date -u -d "2 days ago" +%Y%m%d)
BUF="$REPLAY_DIR/replay_rolling_${TODAY}.parquet"
log "building replay buffer $FROM..$TO (suffix=$SUFFIX, variant=$VARIANT)"
if PYTHONPATH=$REPO $PY -m gx1.scripts.build_online_replay_buffer \
        --from "$FROM" --to "$TO" --suffix "$SUFFIX" \
        --variant "$VARIANT" --out "$BUF" > "$OUT_DIR/buffer_${TODAY}.log" 2>&1; then
    STATUS[buffer]=ok
    log "buffer ok → $BUF"
else
    rc=$?
    # rc=2 (no journals) / rc=3 (no rows) are EXPECTED right after an operating-point
    # suffix change or on quiet weeks — report, don't fail the night.
    STATUS[buffer]="skipped(rc=$rc)"
    log "buffer skipped/failed rc=$rc — see $OUT_DIR/buffer_${TODAY}.log"
fi

# ── 4. KS distribution-drift (advisory) ──────────────────────────────────────
DRIFT_REF=$(PYTHONPATH=$REPO $PY -c "
from gx1_guards.artifacts import load_decision_artifact
from pathlib import Path
p = Path(load_decision_artifact('entry_iql')) / 'drift_reference_v1.parquet'
print(p if p.is_file() else '')" 2>/dev/null || true)
if [[ -n "$DRIFT_REF" ]]; then
    log "KS drift check vs $DRIFT_REF"
    if PYTHONPATH=$REPO $PY -m gx1.audit.feature_liveness \
            --distribution-drift --drift-reference "$DRIFT_REF" \
            --journal-days 7 --strict > "$OUT_DIR/drift_${TODAY}.log" 2>&1; then
        STATUS[drift]=ok
    else
        STATUS[drift]=DRIFT_ALERT
        log "DRIFT-ALERT — see $OUT_DIR/drift_${TODAY}.log (advisory: consider retrain vedtak, never auto)"
    fi
    tail -15 "$OUT_DIR/drift_${TODAY}.log" || true
else
    STATUS[drift]="no_reference"
    log "drift skipped — ACTIVE bundle has no drift_reference_v1.parquet (generate via --write-drift-reference)"
fi

# ── 5. REFIT leg — standing vedtak required (rule 3), PENDING only (rule 8) ──
if [[ -f "$STANDING_VEDTAK_FILE" ]] && [[ -s "$STANDING_VEDTAK_FILE" ]]; then
    VEDTAK=$(head -1 "$STANDING_VEDTAK_FILE" | tr -d '[:space:]')
    if [[ -n "$GIT_DIRTY" ]]; then
        STATUS[refit]="blocked_dirty_tree"
        log "refit BLOCKED: repo tree dirty (rule 2)"
    elif [[ "${STATUS[buffer]}" != "ok" ]]; then
        STATUS[refit]="blocked_no_buffer"
        log "refit BLOCKED: no fresh buffer tonight"
    else
        BASE_BUNDLE=$(PYTHONPATH=$REPO $PY -c "
from gx1_guards.artifacts import load_decision_artifact
print(load_decision_artifact('entry_iql'))")
        CAND_DIR="$ONLINE_IQL_DIR/warmstart_${TODAY}"
        log "REFIT armed (vedtak=$VEDTAK) — warm-start from $(basename "$BASE_BUNDLE")"
        REFIT_OK=true
        for FOLD in FOLD_1 FOLD_2 FOLD_3; do
            PYTHONPATH=$REPO $PY -m gx1.scripts.online_iql_warmstart \
                --base-bundle "$BASE_BUNDLE" --replay "$BUF" \
                --variant "$VARIANT" --fold "$FOLD" \
                --out-dir "$CAND_DIR" --vedtak "$VEDTAK" \
                >> "$OUT_DIR/refit_${TODAY}.log" 2>&1 || { REFIT_OK=false; break; }
        done
        if $REFIT_OK; then
            STATUS[refit]="candidate_PENDING"
            log "candidate written → $CAND_DIR (PENDING — gates + manual contract flip required)"
            # Shadow-score the candidate over the last 2 days of journals (Track-B reuse)
            for d in 1 2; do
                D=$(date -u -d "$d days ago" +%Y%m%d)
                PYTHONPATH=$REPO $PY -m gx1.execution.v12_counterfactual_replay \
                    --journal-date "$D" --journal-suffix "$SUFFIX" \
                    --mode variants --variants auto --bundle-dir "$CAND_DIR" \
                    --out-dir "$OUT_DIR/candidate_shadow" \
                    >> "$OUT_DIR/refit_${TODAY}.log" 2>&1 || true
            done
            log "candidate shadow reports → $OUT_DIR/candidate_shadow/"
        else
            STATUS[refit]=FAIL
            log "refit FAILED — see $OUT_DIR/refit_${TODAY}.log"
        fi
    fi
else
    STATUS[refit]="not_armed"
    log "refit not armed (no standing vedtak at $STANDING_VEDTAK_FILE) — analysis-only night"
fi

# ── 6. Report ─────────────────────────────────────────────────────────────────
{
    echo "{"
    echo "  \"date\": \"$TODAY\","
    echo "  \"suffix\": \"$SUFFIX\","
    echo "  \"variant\": \"$VARIANT\","
    for k in verdicts buffer drift refit; do
        echo "  \"$k\": \"${STATUS[$k]:-unknown}\","
    done
    echo "  \"ram_avail_gb\": $AVAIL_GB"
    echo "}"
} > "$REPORT"
log "nightly report → $REPORT  [verdicts=${STATUS[verdicts]:-?} buffer=${STATUS[buffer]:-?} drift=${STATUS[drift]:-?} refit=${STATUS[refit]:-?}]"
