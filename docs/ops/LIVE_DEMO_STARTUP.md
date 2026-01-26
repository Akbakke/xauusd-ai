# Live DEMO Startup Guide

## Mål
Verifisere end-to-end execution + journaling i ekte markedsdata:
- FARM aktiv i ASIA (natt)
- SNIPER aktiv fra EU/LONDON/NY (morgen)
- Full journaling (entry_snapshot + execution_events + exit_summary)
- Null policy-endringer

## Steg 1 - Preflight Checks

### Environment Variables
```bash
export OANDA_ENV=practice
export OANDA_API_TOKEN=<your_token>
export OANDA_ACCOUNT_ID=<your_account_id>
# I_UNDERSTAND_LIVE_TRADING må IKKE være satt (Practice only)
```

### Verify .env file
```bash
test -f .env && echo "✅ .env exists" || echo "⚠️  .env not found"
```

### Verify Policies
- FARM: `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- SNIPER: `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`

## Steg 2 - Start Prosesser

### Start FARM (ASIA session)
```bash
./scripts/run_live_demo_farm.sh
```

Output directory: `runs/live_demo/FARM_YYYYMMDD_HHMMSS/`

### Start SNIPER (EU/LONDON/NY sessions)
```bash
./scripts/run_live_demo_sniper.sh
```

Output directory: `runs/live_demo/SNIPER_YYYYMMDD_HHMMSS/`

**Note:** SNIPER starter nå, men skal ikke trade i ASIA (session gating aktiv). Blir automatisk aktiv i morgen når EU/London åpner.

## Steg 3 - Preflight Sanity (ved startup)

Begge prosesser logger ved startup:
- Policy/config path + commit hash (baseline)
- Instrument (XAUUSD)
- Account type (Practice)
- Current spread/atr snapshot
- Session gating status
- "TRADING ENABLED" / "NO-TRADE (session)" status

## Steg 4 - Overvåkning i natt

Prosessene logger automatisk:
- Current session
- Open trades count
- Last decision (trade/no-trade + reason)
- Last order status (filled/rejected)

Logs:
- FARM: `runs/live_demo/FARM_*/farm_runtime.log`
- SNIPER: `runs/live_demo/SNIPER_*/sniper_runtime.log`

## Steg 5 - Morgen-sjekk (obligatorisk)

```bash
./scripts/morning_check_live_demo.sh
```

Eller med eksplisitte paths:
```bash
./scripts/morning_check_live_demo.sh \
  --farm-dir runs/live_demo/FARM_YYYYMMDD_HHMMSS \
  --sniper-dir runs/live_demo/SNIPER_YYYYMMDD_HHMMSS
```

Bekrefter:
- SNIPER: 0 trades i natt (ASIA)
- FARM: 0-n trades (ASIA)
- Journal integrity (entry_snapshot, execution_events, exit_summary)

## Steg 6 - I morgen gjennom dagen

La SNIPER kjøre naturlig i sin session (EU/LONDON/NY).

Kjør minst én health check:
```bash
python3 gx1/sniper/analysis/weekly_health_report.py \
  --journal-root runs/live_demo \
  --start-date $(date -u -d "1 day ago" +%Y-%m-%d) \
  --end-date $(date -u +%Y-%m-%d)
```

## Emergency Stop

### Kill-switch
Send SIGTERM eller SIGINT til prosessene:
```bash
pkill -f "run_live_demo_farm"
pkill -f "run_live_demo_sniper"
```

### Graceful shutdown
Prosessene håndterer SIGTERM/SIGINT og stopper gracefully.

## Incident Response

Hvis noe uventet:
1. Følg `docs/ops/INCIDENT_PLAYBOOK.md`
2. Ikke endre policy i panikk
3. Dokumenter incident i logs
4. Stopp prosesser hvis nødvendig

## Verifisering

### Trade Counts
- SNIPER: 0 trades i ASIA (session gating)
- FARM: 0-n trades i ASIA (naturlig variasjon)

### Journal Integrity
Alle trades skal ha:
- `entry_snapshot` (med session, regime, overlays)
- `execution_events` (ORDER_SUBMITTED, ORDER_FILLED, etc.)
- `exit_summary` (for lukkede trades)

### Session Routing
- ASIA → FARM
- EU/LONDON/NY/OVERLAP → SNIPER
- Ingen overlap/conflicts

