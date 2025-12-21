# SNIPER Q1 2025 Replay - Komplett Sammendrag

**Dato:** 2025-12-17  
**Run Tag:** `SNIPER_OBS_Q1_2025_20251217_120004`  
**Status:** ✅ ALLE CHUNKS FERDIG

---

## Executive Summary

SNIPER Q1 2025 replay kjørte med 7 parallelle workers og fullførte uten krasj. Replay feature parity fix fungerer - vol/trend tags beregnes korrekt fra candles i replay mode. Ingen trades ble tatt fordi Stage-0 blokkerer alle entries.

---

## Kjøringsstatistikk

### Chunk Status
- **Chunk 0:** ✅ FERDIG (1,647 logglinjer, 1,584 bars prosessert)
- **Chunk 1:** ✅ FERDIG (1,664 logglinjer, 1,584 bars prosessert)
- **Chunk 2:** ✅ FERDIG (1,758 logglinjer, 1,584 bars prosessert)
- **Chunk 3:** ✅ FERDIG (1,613 logglinjer, 1,584 bars prosessert)
- **Chunk 4:** ✅ FERDIG (1,710 logglinjer, 1,584 bars prosessert)
- **Chunk 5:** ✅ FERDIG (1,652 logglinjer, 1,584 bars prosessert)
- **Chunk 6:** ✅ FERDIG (1,629 logglinjer, 1,584 bars prosessert)

**Total bars prosessert:** ~11,088 (Q1 2025, EU/OVERLAP/US sessions only)

### Trade Resultat
- **Total trades:** 0
- **Årsak:** Stage-0 blokkerer alle entries (100% skip rate)

### Feil Statistikk
- **Session=UNKNOWN errors:** 626 totalt (~4% av entries)
  - Chunk 0: 69 errors
  - Chunk 1: 86 errors
  - Chunk 2: 180 errors (høyest)
  - Chunk 3: 35 errors (lavest)
  - Chunk 4: 131 errors
  - Chunk 5: 74 errors
  - Chunk 6: 51 errors

### Replay Feature Parity
- **Vol/trend tags beregnet:** 14 ganger (2 per chunk - vol_regime og trend_regime)
- **Stage-0 entries evaluert:** ~9,700 totalt
- **Stage-0 skips:** ~9,700 (100% skip rate)

---

## Detaljert Analyse

### Stage-0 Blokkering
Alle entries blir blokkert av Stage-0 filter. Eksempler:
```
[STAGE_0] Skip entry consideration: trend=TREND_NEUTRAL vol=HIGH session=OVERLAP risk=0.000
[STAGE_0] Skip entry consideration: trend=TREND_UP vol=MEDIUM session=EU risk=0.000
[STAGE_0] Skip entry consideration: trend=TREND_NEUTRAL vol=LOW session=US risk=0.000
```

**Distribusjon (chunk 0):**
- Trend: TREND_NEUTRAL (majoritet), TREND_UP (minoritet)
- Vol: HIGH, MEDIUM, LOW (god distribusjon)
- Session: EU, OVERLAP, US (korrekt distribusjon)

### SNIPER Guard Feil
626 `session=UNKNOWN` errors totalt. Dette skjer når `current_row` ikke har session-satt korrekt før guard kjører. De fleste entries har korrekt session, men noen faller gjennom.

### Replay Feature Parity Verifikasjon
✅ **Vol/trend tags beregnes korrekt:**
```
[REPLAY_TAGS] Computed vol_regime from candles: atr14_pct=0.740 -> regime=HIGH (id=2)
[REPLAY_TAGS] Computed trend_regime from candles: ema12=2647.88 ema26=2646.18 -> regime=TREND_NEUTRAL
```

✅ **De fleste entries har korrekt tags:**
- Entries med korrekt tags: ~9,074 (94% av totalt)
- Entries med UNKNOWN: ~626 (6% av totalt)

---

## Konklusjon

### ✅ Suksesser
1. **Replay feature parity fix fungerer:** Vol/trend tags beregnes fra candles i replay mode
2. **Alle chunks fullførte:** Ingen krasj, alle 7 workers fullførte
3. **Session tags settes korrekt:** 94% av entries har korrekt session (EU/OVERLAP/US)
4. **Trend/vol distribusjon:** God variasjon i trend/vol regimes

### ⚠️ Kjente Problemer
1. **Session=UNKNOWN errors:** 626 totalt (~4% av entries)
   - Årsak: `current_row` oppdateres ikke korrekt før SNIPER guard kjører
   - Løsning: Fikse DataFrame oppdatering i `ensure_replay_tags`
2. **0 trades:** Stage-0 blokkerer alle entries
   - Årsak: Stage-0 filter er for strengt for SNIPER policy
   - Løsning: Justere Stage-0 thresholds eller disable for SNIPER

### 🔍 Neste Steg
1. **Fikse session=UNKNOWN errors:**
   - Sørge for at `current_row["session"]` settes direkte fra `policy_state["session"]`
   - Teste at guard får korrekt session

2. **Undersøke Stage-0 blokkering:**
   - Sjekke Stage-0 config for SNIPER policy
   - Vurdere å disable Stage-0 for SNIPER eller justere thresholds

3. **Justere SNIPER policy:**
   - Vurdere å senke `min_prob_long` threshold
   - Vurdere å endre Stage-0 filter for SNIPER

---

## Output Files

- **Run directory:** `gx1/wf_runs/SNIPER_OBS_Q1_2025_20251217_120004/`
- **Chunk logs:** `gx1/wf_runs/SNIPER_OBS_Q1_2025_20251217_120004/parallel_chunks/chunk_*.log`
- **Trade journal:** Ikke generert (0 trades)

---

## Kommandoer for Verifikasjon

```bash
# Sjekk chunk status
ls -lth gx1/wf_runs/SNIPER_OBS_Q1_2025_*/parallel_chunks/chunk_*/chunk_*.log

# Sjekk for UNKNOWN errors
grep -c "session=UNKNOWN" gx1/wf_runs/SNIPER_OBS_Q1_2025_*/parallel_chunks/chunk_*/chunk_*.log

# Sjekk REPLAY_TAGS
grep "REPLAY_TAGS.*Computed" gx1/wf_runs/SNIPER_OBS_Q1_2025_*/parallel_chunks/chunk_*/chunk_*.log

# Sjekk Stage-0 distribusjon
grep "Skip entry consideration" gx1/wf_runs/SNIPER_OBS_Q1_2025_*/parallel_chunks/chunk_0/chunk_0.log | \
  sed 's/.*trend=\([A-Z_]*\).*vol=\([A-Z_]*\).*session=\([A-Z_]*\).*/\1,\2,\3/' | \
  sort | uniq -c | sort -rn
```

---

*Generert: 2025-12-17*

