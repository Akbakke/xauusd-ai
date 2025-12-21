# Komplett Gjennomgang: Q4 Chunk Regime Analysis

**Dato**: 2025-12-20  
**Run**: `SNIPER_OBS_Q4_2025_baseline_20251219_132806`  
**Analyse-script**: `gx1/sniper/analysis/analyze_q4_chunk_regimes.py`

---

## 1. EXECUTIVE SUMMARY

### Hva vi har gjort
- ✅ Opprettet nytt analyse-script som leser direkte fra chunk-level trade journals
- ✅ Analysert 272 Q4 trades fra parallel chunks (uten merge)
- ✅ Klassifisert regime for alle trades
- ✅ Identifisert kritisk problem med trade journaling

### Hva vi har funnet
- ✅ Q4 baseline performance: **60.60 bps EV/trade, 76.8% winrate, 22.35 payoff**
- ⚠️ **KRITISK**: Alle trades mangler `entry_snapshot` → overlay-metadata ikke tilgjengelig
- ⚠️ Alle 272 trades klassifisert som **B_MIXED** (ingen A_TREND eller C_CHOP)

### Hva som gjenstår
- ❌ Overlay-triggering kan ikke verifiseres (mangler `entry_snapshot`)
- ❌ Regime-klassifisering basert på manglende data (alle blir B_MIXED)
- ❌ Session-breakdown ikke mulig (mangler session-data)

---

## 2. DETALJERTE FUNN

### 2.1 Q4 Performance Metrics

| Metric | Verdi |
|--------|-------|
| **Total Trades** | 272 |
| **EV/Trade** | 60.60 bps |
| **Winrate** | 76.8% |
| **Payoff** | 22.35 |
| **P90 Loss** | -0.78 bps |

**Tolkning**: Q4 viser sterk performance med høy winrate og god payoff. Dette er imidlertid basert på kun 272 trades, og alle er klassifisert som B_MIXED (se nedenfor).

### 2.2 Regime Classification

| Regime | Trades | EV/Trade | Winrate | Payoff | P90 Loss |
|--------|--------|----------|---------|--------|----------|
| **B_MIXED** | 272 | 60.60 | 76.8% | 22.35 | -0.78 |

**Kritisk observasjon**: 
- **0 trades** klassifisert som A_TREND
- **0 trades** klassifisert som C_CHOP
- **Alle 272 trades** klassifisert som B_MIXED

**Årsak**: Regime-klassifisering krever `trend_regime`, `vol_regime`, `atr_bps`, og `spread_bps`. Disse feltene er ikke tilgjengelige i trade-filene (se 2.3).

### 2.3 Trade File Structure Analysis

**Hva som finnes i trade-filene**:
- ✅ `trade_id`, `run_tag`, `policy_sha256`, `router_sha256`, `manifest_sha256`
- ✅ `exit_summary` (100% coverage) med:
  - `exit_time`, `exit_price`, `exit_reason`
  - `realized_pnl_bps`, `max_mfe_bps`, `max_mae_bps`, `intratrade_drawdown_bps`
- ❌ `entry_snapshot`: **None** (0% coverage)
- ❌ `feature_context`: **None** (0% coverage)
- ❌ `execution_events`: **Empty list** (0% coverage)

**Hva som mangler**:
- `entry_snapshot` (skal inneholde overlay-metadata)
- `feature_context` (skal inneholde regime-inputs)
- `execution_events` (skal inneholde order-fills)

### 2.4 Overlay Coverage Analysis

| Metric | Verdi |
|--------|-------|
| **Trades with entry_snapshot** | 0 / 272 (0.0%) |
| **Trades with sniper_overlays** | 0 / 272 (0.0%) |
| **Trades with Q4_C_CHOP_SESSION_SIZE overlay** | 0 / 272 (0.0%) |
| **Trades with overlay_applied == True** | 0 / 272 (0.0%) |

**Konsekvens**: Overlay-triggering kan **ikke** verifiseres uten `entry_snapshot`.

---

## 3. ROOT CAUSE ANALYSIS

### 3.1 Problem: entry_snapshot er None

**Symptom**: Alle trade JSON-filer har `entry_snapshot = None`.

**ROOT CAUSE IDENTIFISERT** ✅:

Fra logger (`oanda_demo_runner.log`):
```
[WARNING] [TRADE_JOURNAL] Failed to log structured entry snapshot: 
TradeJournal.log_entry_snapshot() got an unexpected keyword argument 'units'
```

**Årsak**: 
- `entry_manager.py` kaller `log_entry_snapshot()` med `units` og `base_units` parametere
- `trade_journal.py` aksepterer **ikke** disse parametrene i funksjonssignaturen
- Exception oppstår, fanges i `entry_manager.py`, og `entry_snapshot` blir aldri satt

**Verifisert i kode**:
- `entry_manager.py` linje 2802-2803: sender `units=units_out, base_units=base_units`
- `trade_journal.py` linje 230-255: `log_entry_snapshot()` har **ikke** `units` eller `base_units` i signaturen

### 3.2 Verifisering av Trade Journaling Flow

**Fra `entry_manager.py` (linje 2796-2828)**:
```python
self._runner.trade_journal.log_entry_snapshot(
    trade_id=trade.trade_id,
    entry_time=entry_time_iso,
    # ... mange parametere ...
    sniper_overlays=overlays_meta,  # ← Plural
    sniper_overlay=overlays_meta[-1] if overlays_meta else None,  # ← Singular
)
```

**Fra `trade_journal.py` (linje 230-319)**:
```python
def log_entry_snapshot(
    self,
    # ... parametere ...
    sniper_overlay: Optional[Dict[str, Any]] = None,  # ← Bare singular
) -> None:
    # ...
    entry_snapshot: Dict[str, Any] = {
        # ... felter ...
    }
    if sniper_overlay:
        entry_snapshot["sniper_overlay"] = sniper_overlay
    trade_journal["entry_snapshot"] = entry_snapshot
    self._write_trade_json(trade_id)
```

**Observasjon**: 
- `sniper_overlays` (plural) blir sendt, men `log_entry_snapshot` aksepterer bare `sniper_overlay` (singular)
- Dette betyr at bare den siste overlay-en blir lagret, ikke hele listen
- Men dette burde ikke forhindre at `entry_snapshot` blir satt i det hele tatt

### 3.3 Hypotese: Exception i log_entry_snapshot

**Mulig scenario**:
1. `log_entry_snapshot()` kalles
2. Exception oppstår (f.eks. `_get_trade_journal()` feiler)
3. Exception fanges i `except Exception as e:` (linje 318-319)
4. `logger.warning()` logges, men trade fortsetter
5. `entry_snapshot` blir aldri satt

**Verifisering nødvendig**: Sjekk logger for warnings om trade journaling-feil.

---

## 4. KONSEKVENSER

### 4.1 For Overlay Verification

**Problem**: Overlay-triggering kan ikke verifiseres uten `entry_snapshot`.

**Konsekvenser**:
- Kan ikke bekrefte at `Q4_C_CHOP_SESSION_SIZE` overlay faktisk trigges
- Kan ikke verifisere at `overlay_applied == True` for riktige trades
- Kan ikke analysere overlay-impact på trade size

**Løsning**: Fikse trade journaling slik at `entry_snapshot` blir skrevet.

### 4.2 For Regime Classification

**Problem**: Regime-klassifisering basert på manglende data.

**Konsekvenser**:
- Alle trades klassifisert som B_MIXED (default fallback)
- Kan ikke identifisere A_TREND eller C_CHOP trades
- Kan ikke analysere regime-spesifikk performance

**Løsning**: Fikse trade journaling slik at `feature_context` eller `entry_snapshot` inneholder regime-inputs.

### 4.3 For Session Breakdown

**Problem**: Session-data ikke tilgjengelig.

**Konsekvenser**:
- Kan ikke analysere performance per session (EU/OVERLAP/US)
- Kan ikke verifisere session-basert overlay-triggering

**Løsning**: Fikse trade journaling slik at `session` blir lagret i `entry_snapshot`.

---

## 5. NESTE STEG

### 5.1 Umiddelbare handlinger (ikke gjort i denne runden)

1. **✅ ROOT CAUSE IDENTIFISERT**
   - Logger viser: `log_entry_snapshot() got an unexpected keyword argument 'units'`
   - `entry_manager.py` sender `units` og `base_units`, men `trade_journal.py` aksepterer dem ikke

2. **Fikse parameter mismatch (kritisk)**
   - Legg til `units: Optional[int] = None` og `base_units: Optional[int] = None` i `log_entry_snapshot()` signature
   - Lagre `units` og `base_units` i `entry_snapshot` dict
   - Oppdater `log_entry_snapshot()` til å akseptere `sniper_overlays` (plural) i tillegg til `sniper_overlay` (singular)
   - Lagre hele overlay-listen i `entry_snapshot["sniper_overlays"]`, ikke bare den siste

3. **Re-run Q4 baseline etter fix**
   - Verifiser at `entry_snapshot` nå inneholder alle nødvendige felter
   - Verifiser at overlay-metadata er tilgjengelig

### 5.2 Fremtidige analyser (når trade journaling er fikset)

1. **Re-run Q4 baseline replay**
   - Med fikset trade journaling
   - Verifiser at `entry_snapshot` nå inneholder overlay-metadata

2. **Re-run chunk-level analysis**
   - Med komplette trade journals
   - Verifiser overlay-triggering
   - Analyser regime-spesifikk performance

3. **Session-based analysis**
   - Når session-data er tilgjengelig
   - Analyser performance per session
   - Verifiser session-basert overlay-triggering

---

## 6. KONKLUSJON

### Hva vi vet
- ✅ Q4 baseline viser sterk performance (60.60 bps EV, 76.8% winrate)
- ✅ Analyse-script fungerer korrekt (leser fra chunks, klassifiserer regime)
- ⚠️ Trade journaling bevarer ikke `entry_snapshot`

### Hva vi ikke vet
- ❌ Om overlay faktisk trigges (mangler metadata)
- ❌ Regime-distribusjon (alle er B_MIXED pga. manglende data)
- ❌ Session-basert performance (mangler session-data)

### Hva som må fikses
1. **Trade journaling** må bevare `entry_snapshot` med overlay-metadata
2. **Parameter mismatch** må fikses (`sniper_overlays` vs `sniper_overlay`)
3. **Re-run** Q4 baseline når trade journaling er fikset

### Status
- ✅ **Steg 1**: Chunk-level sannhetsanalyse - **FERDIG**
- ✅ **Steg 2**: Verifiser overlay-trigger - **IKKE MULIG** (mangler data)
- ⏸️ **Steg 3**: Ikke gjort (som planlagt)

**Rapport generert**: `reports/SNIPER_Q4_CHUNK_REGIME_ANALYSIS__20251220_103644.md`

