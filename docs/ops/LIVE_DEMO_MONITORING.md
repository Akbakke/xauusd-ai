# Live Demo Monitoring Commands

**Purpose:** Quick reference for monitoring FARM and SNIPER live-demo processes.

**Last Updated:** 2025-12-22

---

## Quick Status Check

```bash
# Find latest FARM log
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
echo "FARM Log: $FARM_LOG"

# Find latest SNIPER log
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
echo "SNIPER Log: $SNIPER_LOG"
```

---

## 1. Process Health

### Check if Process is Running

```bash
# FARM
pgrep -f "run_live_demo_farm" && echo "FARM: RUNNING" || echo "FARM: NOT RUNNING"

# SNIPER
pgrep -f "run_live_demo_sniper" && echo "SNIPER: RUNNING" || echo "SNIPER: NOT RUNNING"
```

### Check Process Age

```bash
# FARM
ps -p $(pgrep -f "run_live_demo_farm" | head -1) -o etime,cmd 2>/dev/null || echo "FARM not running"

# SNIPER
ps -p $(pgrep -f "run_live_demo_sniper" | head -1) -o etime,cmd 2>/dev/null || echo "SNIPER not running"
```

---

## 2. Cycle Status

### Recent Cycles

```bash
# FARM - last 5 cycles
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
grep "Cycle complete" "$FARM_LOG" | tail -5

# SNIPER - last 5 cycles
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep "Cycle complete" "$SNIPER_LOG" | tail -5
```

### Cycle Count (Last Hour)

```bash
# FARM
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
grep "Cycle complete" "$FARM_LOG" | grep "$(date -u +%Y-%m-%d)" | wc -l

# SNIPER
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep "Cycle complete" "$SNIPER_LOG" | grep "$(date -u +%Y-%m-%d)" | wc -l
```

### SNIPER Cycle Details

```bash
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep "\[SNIPER_CYCLE\]" "$SNIPER_LOG" | tail -5
```

---

## 3. Exceptions and Errors

### Critical Errors

```bash
# FARM
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
grep -i "error\|exception\|traceback\|fatal" "$FARM_LOG" | tail -10

# SNIPER
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep -i "error\|exception\|traceback\|fatal" "$SNIPER_LOG" | tail -10
```

### Parity Enabled Errors (Should be 0)

```bash
# FARM
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
echo "AttributeError.*parity_enabled: $(grep -c 'AttributeError.*parity_enabled' "$FARM_LOG" 2>/dev/null || echo 0)"
echo "Health signal.*parity_enabled: $(grep -c 'Health signal update failed.*parity_enabled' "$FARM_LOG" 2>/dev/null || echo 0)"

# SNIPER
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
echo "AttributeError.*parity_enabled: $(grep -c 'AttributeError.*parity_enabled' "$SNIPER_LOG" 2>/dev/null || echo 0)"
echo "Health signal.*parity_enabled: $(grep -c 'Health signal update failed.*parity_enabled' "$SNIPER_LOG" 2>/dev/null || echo 0)"
```

### OANDA API Errors

```bash
# FARM
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
grep -i "oanda.*error\|401\|403\|timeout" "$FARM_LOG" | tail -5

# SNIPER
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep -i "oanda.*error\|401\|403\|timeout" "$SNIPER_LOG" | tail -5
```

---

## 4. Order Events

### Order Creation

```bash
# FARM
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
grep -i "order.*create\|order.*open\|tradeOpened" "$FARM_LOG" | tail -10

# SNIPER
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep -i "order.*create\|order.*open\|tradeOpened" "$SNIPER_LOG" | tail -10
```

### Order Fills/Rejections

```bash
# FARM
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
grep -i "filled\|rejected\|tradeClosed" "$FARM_LOG" | tail -10

# SNIPER
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep -i "filled\|rejected\|tradeClosed" "$SNIPER_LOG" | tail -10
```

---

## 5. Trade Journal

### New Trade Files

```bash
# FARM
FARM_DIR=$(ls -1dt runs/live_demo/FARM_* | head -1)
if [ -d "$FARM_DIR/trade_journal/trades" ]; then
    echo "FARM trades: $(ls -1 "$FARM_DIR/trade_journal/trades"/*.json 2>/dev/null | wc -l)"
    echo "Latest trade: $(ls -1t "$FARM_DIR/trade_journal/trades"/*.json 2>/dev/null | head -1)"
else
    echo "FARM: No trade journal directory"
fi

# SNIPER
SNIPER_DIR=$(ls -1dt runs/live_demo/SNIPER_* | head -1)
if [ -d "$SNIPER_DIR/trade_journal/trades" ]; then
    echo "SNIPER trades: $(ls -1 "$SNIPER_DIR/trade_journal/trades"/*.json 2>/dev/null | wc -l)"
    echo "Latest trade: $(ls -1t "$SNIPER_DIR/trade_journal/trades"/*.json 2>/dev/null | head -1)"
else
    echo "SNIPER: No trade journal directory"
fi
```

### Latest Trade Summary

```bash
# FARM
FARM_DIR=$(ls -1dt runs/live_demo/FARM_* | head -1)
LATEST_TRADE=$(ls -1t "$FARM_DIR/trade_journal/trades"/*.json 2>/dev/null | head -1)
if [ -n "$LATEST_TRADE" ]; then
    echo "FARM Latest Trade:"
    jq -r '.entry_snapshot.session, .entry_snapshot.base_units, .exit_summary.pnl_bps // "N/A"' "$LATEST_TRADE" 2>/dev/null
fi

# SNIPER
SNIPER_DIR=$(ls -1dt runs/live_demo/SNIPER_* | head -1)
LATEST_TRADE=$(ls -1t "$SNIPER_DIR/trade_journal/trades"/*.json 2>/dev/null | head -1)
if [ -n "$LATEST_TRADE" ]; then
    echo "SNIPER Latest Trade:"
    jq -r '.entry_snapshot.session, .entry_snapshot.base_units, .exit_summary.pnl_bps // "N/A"' "$LATEST_TRADE" 2>/dev/null
fi
```

---

## 6. Session and Gating

### Current Session

```bash
# FARM - ASIA session
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
grep -i "session\|ASIA" "$FARM_LOG" | tail -3

# SNIPER - EU/LONDON/NY/OVERLAP
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep "\[SNIPER_CYCLE\]" "$SNIPER_LOG" | tail -3 | grep -o "session=[^ ]*"
```

### Gating Status

```bash
# SNIPER
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep "\[SNIPER_CYCLE\]" "$SNIPER_LOG" | tail -3 | grep -o "in_scope=[^ ]*"
```

---

## 7. Boot Status

### Boot Messages

```bash
# FARM
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log | head -1)
grep -i "boot\|start\|preflight\|parity_enabled" "$FARM_LOG" | head -10

# SNIPER
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log | head -1)
grep -i "boot\|start\|preflight\|parity_enabled" "$SNIPER_LOG" | head -10
```

---

## 8. All-in-One Status Check

```bash
#!/bin/bash
# Quick status check for both FARM and SNIPER

echo "=== FARM Status ==="
FARM_LOG=$(ls -1dt runs/live_demo/FARM_*/farm_runtime.log 2>/dev/null | head -1)
if [ -n "$FARM_LOG" ]; then
    echo "Log: $FARM_LOG"
    echo "Running: $(pgrep -f 'run_live_demo_farm' > /dev/null && echo 'YES' || echo 'NO')"
    echo "Cycles (last hour): $(grep 'Cycle complete' "$FARM_LOG" | grep "$(date -u +%Y-%m-%d)" | wc -l)"
    echo "Parity errors: $(grep -c 'AttributeError.*parity_enabled\|Health signal.*parity_enabled' "$FARM_LOG" 2>/dev/null || echo 0)"
else
    echo "No FARM log found"
fi

echo ""
echo "=== SNIPER Status ==="
SNIPER_LOG=$(ls -1dt runs/live_demo/SNIPER_*/sniper_runtime.log 2>/dev/null | head -1)
if [ -n "$SNIPER_LOG" ]; then
    echo "Log: $SNIPER_LOG"
    echo "Running: $(pgrep -f 'run_live_demo_sniper' > /dev/null && echo 'YES' || echo 'NO')"
    echo "Cycles (last hour): $(grep 'Cycle complete' "$SNIPER_LOG" | grep "$(date -u +%Y-%m-%d)" | wc -l)"
    echo "Parity errors: $(grep -c 'AttributeError.*parity_enabled\|Health signal.*parity_enabled' "$SNIPER_LOG" 2>/dev/null || echo 0)"
else
    echo "No SNIPER log found"
fi
```

---

**Note:** All commands assume you are in the project root directory (`/Users/andrekildalbakke/Desktop/GX1 XAUUSD`).

