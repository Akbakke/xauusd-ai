#!/usr/bin/env bash
# Wait for V6 disk-trainer (PID arg) to exit, sanity-check the resulting
# bundle, then run the V5/V6 head-to-head test eval. Runs unattended.
#
# Usage: post_v6_pipeline.sh <V6_TRAINER_PID> <V5_BUNDLE_DIR>
#
# Logs to /tmp/post_v6_pipeline.log. Bundle path is auto-detected as the
# most recent EXIT_V6_DISK__BIDIR_2026Q2_* directory.

set -uo pipefail

V6_PID="${1:?usage: post_v6_pipeline.sh <PID> <V5_BUNDLE>}"
V5_BUNDLE="${2:?usage: post_v6_pipeline.sh <PID> <V5_BUNDLE>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/andre2/venvs/gx1/bin/python3
export GX1_DATA=/home/andre2/GX1_DATA

log() { echo "[post-v6 $(date -u +%H:%M:%S)] $*"; }

log "watching PID $V6_PID — V5 reference: $V5_BUNDLE"
while kill -0 "$V6_PID" 2>/dev/null; do
    sleep 60
done
log "V6 trainer exited"

V6_BUNDLE=$(ls -dt /home/andre2/GX1_DATA/models/exit_transformer_v0/EXIT_V6_DISK__BIDIR_2026Q2_* 2>/dev/null | head -1)
if [[ -z "$V6_BUNDLE" ]]; then
    log "FAIL: no EXIT_V6_DISK bundle found — V6 trainer may have crashed before saving"
    log "       check /tmp/v6_disk_train.log for the exit reason"
    exit 1
fi
log "V6 bundle: $V6_BUNDLE"

# Sanity check files
for f in exit_transformer_v0.pt transformer_config.json manifest.json; do
    if [[ ! -f "$V6_BUNDLE/$f" ]]; then
        log "FAIL: missing $V6_BUNDLE/$f"
        exit 1
    fi
done
log "bundle file structure OK"

log "=== sanity-check manifest ==="
"$PYTHON" - <<PYEOF
import json, sys
m = json.load(open("$V6_BUNDLE/manifest.json"))
best = m.get("best_val_loss")
hist = m.get("history") or []
test = m.get("test_metrics") or {}
test_acc = test.get("acc_at_05")
test_n = test.get("n")
test_loss_main = test.get("loss_main")
n_train = m.get("n_train")
n_val = m.get("n_val")
n_test = m.get("n_test")

problems = []
if best is None or (best != best) or best > 1e6:
    problems.append(f"best_val_loss invalid: {best}")
if test_acc is None or test_acc < 0.5:
    problems.append(f"test acc_at_05 too low: {test_acc}")
if not hist:
    problems.append("history is empty")
elif len(hist) < 2:
    problems.append(f"history has only {len(hist)} epoch(s) — early failure?")

if problems:
    print("[sanity] FAIL")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)

print(f"[sanity] OK")
print(f"  bundle             : {m['bundle_name']}")
print(f"  best_val_loss      : {best:.4f}")
print(f"  test acc_at_05     : {test_acc:.4f}")
print(f"  test loss_main     : {test_loss_main:.4f} (n={test_n:,})")
print(f"  epochs trained     : {len(hist)}")
print(f"  n train/val/test   : {n_train:,} / {n_val:,} / {n_test:,}")
training = m.get("training") or {}
print(f"  side_balance       : {training.get('side_balance_weight')}")
print(f"  ema_decay          : {training.get('ema_decay')}")
print(f"  mixed_precision    : {training.get('mixed_precision')}")
print(f"  focal_gamma        : {training.get('focal_gamma')}")

last_val = hist[-1].get("val") or {}
print(f"  last epoch val_main: {last_val.get('loss_main')}")
print(f"  last epoch val_acc : {last_val.get('acc_at_05')}")
PYEOF
SANITY_RC=$?
if [[ $SANITY_RC -ne 0 ]]; then
    log "sanity check FAILED — skipping head-to-head eval"
    exit $SANITY_RC
fi

log "=== V5 vs V6 head-to-head test eval ==="
bash "$SCRIPT_DIR/compare_v5_v6_bundles.sh" "$V5_BUNDLE" "$V6_BUNDLE" test
log "DONE"
