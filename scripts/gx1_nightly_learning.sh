#!/usr/bin/env bash
# GX1 nightly continuous-learning loop — ladder wave (user vedtak «Ja til alle 3» 2026-06-12).
#
# NIGHTLY legs (no vedtak needed; analysis + idempotent data maintenance):
#   1.  RAM guard (abort <8 GB available — AGENTS.md OOM hard ceiling)
#   2.  Accumulate per-trade verdicts (trade_verdicts_*.jsonl → regret_dataset.parquet)
#   2b. Canonical tape freshener — WRITES the M1/M5 canonical tapes (idempotent
#       append+dedupe; OANDA history, complete candles only; fail-loud chunk guard)
#   3.  ENTRY replay buffer (matured D-8..D-2; 25h K-aging) + EXIT-side transition
#       buffer (dataset-only — refit rewards go through the exit builder's defs)
#   4.  KS distribution-drift check (rule-9 drift leg — ADVISORY, never blocks)
#   5.  Nightly report JSON in nightly_learning/ + severity-folded exit code
#
# REFIT leg (CLAUDE.md rule 3 — NEVER runs without a standing vedtak):
#   Armed ONLY when $STANDING_VEDTAK_FILE exists and contains a vedtak id.
#   Then: 3-fold warm-start refit (anti-forgetting cement mix when the ACTIVE
#   bundle carries cement_replay_sample_v1.parquet) → PENDING candidate under
#   online_iql/ → D-1 out-of-sample shadow report → quick candidate gate →
#   shadow rotation ONLY on gate PASS. The candidate stays PENDING — this script
#   NEVER flips PROJECT_STATE_artifacts.json (rule 8: promote is a manual flip).
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

# ── 2b. Canonical tape freshener (M1 → M5; ladder wave 2026-06-12) ───────────
# Both canonical tapes froze at the Jun 8 repair — nothing scheduled freshened
# them. Chain: OANDA history (complete candles only) → v12_backfill_to_present
# (per-year dedupe keep='last', idempotent, no-op if gap<5min) → m1_to_m5
# downsample (drops <5-bar buckets: a partial live-edge bucket can never be
# emitted; 1h overlap self-heals the edge). Single-writer: never run twice
# concurrently. Bonus: OANDA history FILLS registered machine-down gaps the
# live collector missed (06-10 OOM / 06-11 reboot).
log "freshening canonical tapes (M1 backfill → M5 downsample)"
TAPE_OK=true
PYTHONPATH=$REPO $PY $REPO/gx1/execution/v12_backfill_to_present.py \
    > "$OUT_DIR/tape_m1_${TODAY}.log" 2>&1 || TAPE_OK=false
# A failed 1-day fetch chunk is skipped-and-continued upstream and becomes a
# PERMANENT hole (next run starts past it) — fail the step LOUD instead.
if grep -qE "chunk [0-9]+ failed" "$OUT_DIR/tape_m1_${TODAY}.log"; then
    TAPE_OK=false
    log "TAPE M1 backfill had FAILED chunks — permanent-hole risk; repair before next run"
fi
if $TAPE_OK; then
    PYTHONPATH=$REPO $PY $REPO/gx1/execution/v12_m1_to_m5_downsample.py \
        > "$OUT_DIR/tape_m5_${TODAY}.log" 2>&1 || TAPE_OK=false
fi
if $TAPE_OK && PYTHONPATH=$REPO $PY - >> "$OUT_DIR/tape_m5_${TODAY}.log" 2>&1 <<'PYEOF'
import sys, pathlib, pandas as pd
root = pathlib.Path("/home/andre2/GX1_DATA/data/oanda/canonical")
last = lambda tape: max(pd.to_datetime(pd.read_parquet(f, columns=["time"])["time"], utc=True).max()
                        for f in sorted((root / tape).glob("year=*/part-000.parquet")))
m1, m5 = last("xauusd_m1_bid_ask__CANONICAL"), last("xauusd_m5_bid_ask__CANONICAL")
lag_h = (pd.Timestamp.now(tz="UTC") - m1).total_seconds() / 3600
print(f"[TAPE_FRESHEN] m1_last={m1} m5_last={m5} m1_lag_h={lag_h:.1f}")
# 36h allows the Sunday 03:30Z run (~30.6h back to Friday close); M5 within the
# downsampler's own 1h self-heal window of M1.
sys.exit(0 if lag_h < 36 and (m1 - m5).total_seconds() <= 3600 else 1)
PYEOF
then
    STATUS[tape]=ok
    log "tapes fresh (see tape_m5_${TODAY}.log for cutoffs)"
else
    STATUS[tape]=FAIL
    log "tape freshener FAILED — see $OUT_DIR/tape_m1_${TODAY}.log / tape_m5_${TODAY}.log"
fi

# ── 3. Rolling live-regret replay buffers (matured days only) ─────────────────
FROM=$(date -u -d "8 days ago" +%Y%m%d)
TO=$(date -u -d "2 days ago" +%Y%m%d)
BUF="$REPLAY_DIR/replay_rolling_${TODAY}.parquet"
log "building ENTRY replay buffer $FROM..$TO (suffix=$SUFFIX, variant=$VARIANT)"
if PYTHONPATH=$REPO $PY -m gx1.scripts.build_online_replay_buffer \
        --from "$FROM" --to "$TO" --suffix "$SUFFIX" \
        --variant "$VARIANT" --out "$BUF" > "$OUT_DIR/buffer_${TODAY}.log" 2>&1; then
    STATUS[buffer]=ok
    log "entry buffer ok → $BUF"
else
    rc=$?
    # rc=2 (no journals) / rc=3 (no rows) are EXPECTED right after an operating-point
    # suffix change or on quiet weeks — report, don't fail the night.
    STATUS[buffer]="skipped(rc=$rc)"
    log "entry buffer skipped/failed rc=$rc — see $OUT_DIR/buffer_${TODAY}.log"
fi
# EXIT-side transitions (ladder wave 2026-06-12): per-(trade, M1-bar) 209-dim
# states + regret labels. Dataset only — exit-IQL refit rewards go through the
# exit builder's one-truth defs (see build_online_exit_replay_buffer header).
EXIT_BUF="$REPLAY_DIR/exit_replay_rolling_${TODAY}.parquet"
if PYTHONPATH=$REPO $PY -m gx1.scripts.build_online_exit_replay_buffer \
        --from "$FROM" --to "$TO" --suffix "$SUFFIX" \
        --out "$EXIT_BUF" > "$OUT_DIR/exit_buffer_${TODAY}.log" 2>&1; then
    STATUS[exit_buffer]=ok
    log "exit buffer ok → $EXIT_BUF"
else
    rc=$?
    STATUS[exit_buffer]="skipped(rc=$rc)"
    log "exit buffer skipped/failed rc=$rc — see $OUT_DIR/exit_buffer_${TODAY}.log"
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
        # Timestamped dir (2026-06-12 adversarial finding): a date-only key let a
        # same-day re-run leave a FOLD-mixed chimera (new FOLD_1 + stale FOLD_2/3)
        # indistinguishable to the gate's sanity step.
        CAND_DIR="$ONLINE_IQL_DIR/warmstart_${TODAY}T$(date -u +%H%M%S)"
        log "REFIT armed (vedtak=$VEDTAK) — warm-start from $(basename "$BASE_BUNDLE")"
        # Anti-forgetting mix (ladder wave 2026-06-12): anchor the refit on a frozen
        # 20k cement-distribution sample (states+rewards, posgate-parity) living in
        # the ACTIVE bundle dir. Contract-resolved $BASE_BUNDLE means a bundle flip
        # automatically points at THAT bundle's sample; if the new bundle has none
        # yet, run unmixed + warn loud (re-export per build_online_replay_buffer
        # --export-cement-sample) rather than kill the night.
        MIX_ARGS=()
        CEMENT_SAMPLE="$BASE_BUNDLE/cement_replay_sample_v1.parquet"
        if [[ -s "$CEMENT_SAMPLE" ]]; then
            MIX_ARGS=(--mix-cement "$CEMENT_SAMPLE" --mix-weight 0.5)
        else
            log "WARNING: no cement_replay_sample_v1.parquet in $(basename "$BASE_BUNDLE") — refit runs UNMIXED (forgetting risk)"
        fi
        REFIT_OK=true
        for FOLD in FOLD_1 FOLD_2 FOLD_3; do
            PYTHONPATH=$REPO $PY -m gx1.scripts.online_iql_warmstart \
                --base-bundle "$BASE_BUNDLE" --replay "$BUF" \
                --variant "$VARIANT" --fold "$FOLD" \
                "${MIX_ARGS[@]}" \
                --out-dir "$CAND_DIR" --vedtak "$VEDTAK" \
                >> "$OUT_DIR/refit_${TODAY}.log" 2>&1 || { REFIT_OK=false; break; }
        done
        if $REFIT_OK; then
            STATUS[refit]="candidate_PENDING"
            log "candidate written → $CAND_DIR (PENDING — gates + manual contract flip required)"
            # Shadow-score the candidate on D-1 ONLY — D-2 sits INSIDE its own
            # training window (buffer D-8..D-2) and would read as out-of-sample
            # evidence when it is not (2026-06-12 adversarial finding).
            D=$(date -u -d "1 day ago" +%Y%m%d)
            PYTHONPATH=$REPO $PY -m gx1.execution.v12_counterfactual_replay \
                --journal-date "$D" --journal-suffix "$SUFFIX" \
                --mode variants --variants auto --bundle-dir "$CAND_DIR" \
                --out-dir "$OUT_DIR/candidate_shadow" \
                >> "$OUT_DIR/refit_${TODAY}.log" 2>&1 || true
            log "candidate shadow report (D-1, out-of-sample) → $OUT_DIR/candidate_shadow/"
            # Auto-gate evidence (ladder wave 2026-06-12): quick candidate gate run —
            # PASS/FAIL verdict json under nightly_learning/candidate_gates/.
            # Evidence only; promote stays a manual contract flip (rule 8).
            if bash $REPO/scripts/gx1_candidate_gate.sh "$CAND_DIR" --quick \
                    >> "$OUT_DIR/candidate_gate_${TODAY}.log" 2>&1; then
                STATUS[gate]=PASS
                log "candidate gate PASS (verdict in nightly_learning/candidate_gates/)"
                # Rotate the in-process shadow ONLY on gate PASS — a failed candidate
                # must not evict a passing one from the single shadow slot. The runner
                # reads this file at LAUNCH — picked up at its next restart.
                echo "$CAND_DIR" > /home/andre2/GX1_DATA/config/shadow_bundle_dir.txt
                log "in-process shadow rotated → $CAND_DIR (takes effect at next runner restart)"
            else
                STATUS[gate]=FAIL
                log "candidate gate FAIL/blocked — shadow NOT rotated; see $OUT_DIR/candidate_gate_${TODAY}.log"
            fi
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
    for k in verdicts tape buffer exit_buffer drift refit gate; do
        echo "  \"$k\": \"${STATUS[$k]:-unknown}\","
    done
    echo "  \"ram_avail_gb\": $AVAIL_GB"
    echo "}"
} > "$REPORT"
log "nightly report → $REPORT  [verdicts=${STATUS[verdicts]:-?} tape=${STATUS[tape]:-?} buffer=${STATUS[buffer]:-?} exit_buf=${STATUS[exit_buffer]:-?} drift=${STATUS[drift]:-?} refit=${STATUS[refit]:-?} gate=${STATUS[gate]:-?}]"

# Severity fold (2026-06-12 adversarial finding): the oneshot must not report
# SUCCESS on a wasted night. Hard failures = nonzero so journalctl/OnFailure can
# alert; skipped(rc=…) buffers and not_armed refit remain a successful night.
if [[ "${STATUS[tape]:-}" == "FAIL" || "${STATUS[refit]:-}" == "FAIL" || "${STATUS[verdicts]:-}" == "FAIL" ]]; then
    exit 1
fi
exit 0
