#!/usr/bin/env bash
# V12 prebuilt-refresh daemon — keeps canonical_v3 + BASE34 prebuilts fresh
# so the paper-runner's cutoff stays close to real-time.
#
# The runner reads features from these prebuilts:
#   - canonical_v3 prebuilt  (M5 cadence, 92 features)
#   - BASE34 prebuilt        (M1 cadence, 32 augmented features)
#
# Without periodic refresh, cutoff drifts further behind real-time as new
# market bars arrive. This daemon rebuilds both every $INTERVAL_SEC seconds
# (default: 900 = 15 min).
#
# Pipeline per cycle:
#   1. v12_backfill_to_present.py     M1 backfill (5s if up-to-date, ~10 min if huge gap)
#   2. v12_m1_to_m5_downsample.py     M5 extension (~5s)
#   3. canonical_v2 builder            full rebuild (~3-5 min)
#   4. sync canonical_v2 to prebuilt path
#   5. canonical_v3 augment            (~1-2 min)
#   6. build_truth_monday_week_prebuilt_extension_v1   BASE34 M1-rebuild (~5-10 min)
#   7. update manifest                 atomic
#
# Total cycle: ~15-25 min full rebuild. For an incremental builder (which
# only processes new bars) see future work — for now full rebuild is the
# correct-by-construction approach.
#
# Run as background daemon:
#   nohup bash gx1/execution/v12_prebuilt_refresh_daemon.sh > /tmp/refresh_daemon.log 2>&1 &
#
# Stop:
#   pkill -f v12_prebuilt_refresh_daemon.sh

set -uo pipefail
INTERVAL_SEC=${INTERVAL_SEC:-1800}   # 30 min default
REPO=/home/andre2/src/GX1_ENGINE
LOG_DIR=/home/andre2/GX1_DATA/reports/v12_live_data/logs
MANIFEST=/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json
mkdir -p "$LOG_DIR"

cycle=0
while true; do
    cycle=$((cycle + 1))
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    LOG="$LOG_DIR/refresh_daemon_cycle_${cycle}_${TS}.log"
    echo "[refresh-daemon] cycle $cycle starting at $TS" | tee -a "$LOG"

    # Step 1: M1 backfill
    PYTHONPATH=$REPO timeout 600 python3 -u $REPO/gx1/execution/v12_backfill_to_present.py >> "$LOG" 2>&1 \
        && echo "[refresh] ok m1_backfill" | tee -a "$LOG" \
        || echo "[refresh] WARN m1_backfill failed" | tee -a "$LOG"

    # Step 2: M5 downsample (incremental)
    PYTHONPATH=$REPO timeout 120 python3 -u $REPO/gx1/execution/v12_m1_to_m5_downsample.py >> "$LOG" 2>&1 \
        && echo "[refresh] ok m5_downsample" | tee -a "$LOG" \
        || echo "[refresh] WARN m5_downsample failed" | tee -a "$LOG"

    # Step 3: canonical_v2 build
    PYTHONPATH=$REPO timeout 600 python3 -u $REPO/gx1/scripts/materialize_build_canonical_features_v2.py >> "$LOG" 2>&1 \
        && echo "[refresh] ok canonical_v2" | tee -a "$LOG" \
        || { echo "[refresh] FAIL canonical_v2 — skipping rest of cycle" | tee -a "$LOG"; sleep "$INTERVAL_SEC"; continue; }

    # Step 4: sync canonical_v2 to prebuilt path (script writes to reports/, builder reads from data/)
    cp /home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V2/canonical_features_v2.parquet \
       /home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V2_PREBUILT/xauusd_m5_CANONICAL_V2_2020_2026.parquet \
       && echo "[refresh] ok canonical_v2 sync" | tee -a "$LOG"

    # Step 5: canonical_v3 augment
    PYTHONPATH=$REPO timeout 300 python3 -u $REPO/gx1/scripts/materialize_canonical_v3_augment.py >> "$LOG" 2>&1 \
        && echo "[refresh] ok canonical_v3" | tee -a "$LOG" \
        || { echo "[refresh] FAIL canonical_v3 — skipping rest" | tee -a "$LOG"; sleep "$INTERVAL_SEC"; continue; }

    # Step 6: BASE34 M1-rebuild (full from 2020-11-09 to tomorrow-exclusive)
    END_EXCLUSIVE=$(date -u -d 'tomorrow' +%Y-%m-%dT00:00:00Z)
    PYTHONPATH=$REPO timeout 1800 python3 -u $REPO/gx1/scripts/build_truth_monday_week_prebuilt_extension_v1.py \
        --start 2020-11-09T00:00:00Z \
        --end-exclusive "$END_EXCLUSIVE" \
        --output-root /home/andre2/GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES \
        >> "$LOG" 2>&1 \
        && echo "[refresh] ok base34" | tee -a "$LOG" \
        || { echo "[refresh] FAIL base34 — skipping rest" | tee -a "$LOG"; sleep "$INTERVAL_SEC"; continue; }

    # Step 7: promote new BASE34 (atomic manifest update)
    NEW_DIR=$(ls -dt /home/andre2/GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES/monday_week_prebuilt_extension_* | head -1)
    NEW_FILE=$(ls "$NEW_DIR"/xauusd_m1_EXPANDED_BASE34_CTX16CAT6_*_RAW_INDEX.parquet 2>/dev/null | head -1)
    if [[ -f "$NEW_FILE" ]]; then
        cp "$MANIFEST" "${MANIFEST}.backup_$(date -u +%Y%m%dT%H%M%SZ)"
        python3 -c "
import json, hashlib
import pyarrow.parquet as pq
fp = '$NEW_FILE'
sha = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
rows = pq.ParquetFile(fp).metadata.num_rows
cols = len(pq.read_schema(fp).names)
m = {
    'cols_total': cols, 'created_utc': '$TS', 'kind': 'BASE28_CANONICAL_MANIFEST',
    'note': 'refresh-daemon cycle $cycle',
    'parquet_path': fp, 'parquet_sha256': sha, 'rows': rows,
}
json.dump(m, open('$MANIFEST', 'w'), indent=2)
" && echo "[refresh] ok manifest updated → $NEW_FILE" | tee -a "$LOG"
    else
        echo "[refresh] WARN no new base34 file found in $NEW_DIR" | tee -a "$LOG"
    fi

    DURATION=$(($(date +%s) - $(date -d "$TS" +%s) ))
    echo "[refresh-daemon] cycle $cycle done in ${DURATION}s — sleeping $INTERVAL_SEC s" | tee -a "$LOG"
    sleep "$INTERVAL_SEC"
done
