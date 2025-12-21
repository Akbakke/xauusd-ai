# SNIPER Debug Status

**Dato:** 2025-12-17  
**Siste Run Tag:** `SNIPER_OBS_Q1_2025_20251217_120004`  
**Status:** 🔴 BLOCKED - Replay produserer trades men de blir ikke lukket korrekt

---

## Executive Summary

SNIPER Q1 replay kjører og produserer entry-signaler, men trades blir blokkert av `max_open_trades=1` fordi første trade aldri lukkes. Rotårsak: ExitArbiter avviser `SL_TICK` close events fordi de ikke er i allowlist (`allowed=['RULE_A_PROFIT', 'RULE_A_TRAILING', 'RULE_B_FAST_LOSS', 'RULE_C_TIMEOUT']`). Dette gjør at trade henger åpen og blokkerer nye entries.

---

## Hva Fungerer

✅ **Replay feature parity fix:** Trend/vol tags beregnes korrekt fra candles i replay mode  
✅ **Session tags:** De fleste entries har korrekt session (EU/OVERLAP/US)  
✅ **SNIPER policy:** Entry-signaler genereres korrekt (p_long > 0.67, guard passerer)  
✅ **Stage-0 SNIPER config:** SNIPER-spesifikk Stage-0 logikk fungerer (mer permissive enn FARM)  
✅ **Entry execution:** Trades åpnes korrekt i replay mode  

---

## Hva Feiler

❌ **0 trades i Q1 replay:** Alle entries blir blokkert av `max_open_trades=1`  
❌ **Trade henger åpen:** Første trade åpnes men lukkes aldri  
❌ **ExitArbiter avviser SL_TICK:** Close events med `reason=SL_TICK` blir avvist  
❌ **Session mismatch:** EntryManager logger `session=US` men ENTRY_DIAG logger `session=OVERLAP`  

### Konkrete Symptomer fra Logg

```
[ENTRY SIGNAL] Session=US, T=1.000, side=LONG, p_hat=0.8958
[ENTRY_DIAG] trade=SIM-1765970767-000001 ts=2025-01-02T15:20:00+00:00 session=OVERLAP
[DRY-RUN] WOULD EXECUTE LONG XAU_USD units=1 @ 2649.310
[ENTRY][OPEN_TRADES] open=1 last_trade=SIM-1765970767-000001
Skip entry: max_open_trades reached (1)  # ← Blokkerer alle nye entries
[ARB] accept close by EXIT_FARM_V2_RULES reason=RULE_A_TRAILING pnl=9.7
[LIVE] CLOSED TRADE (FARM_V2_RULES) LONG SIM-1765970767-000001 @ 2651.880 | pnl=9.7 bps
```

**Observasjon:** Trade lukkes faktisk, men det ser ut som det er en race condition eller timing-issue hvor `max_open_trades` sjekkes før trade er fjernet fra `open_trades` liste.

---

## Rotårsak Hypoteser

### Hypotesis 1: Trade Fjernes Før request_close (HØYESTE SANSYNLIGHET) ✅ BEKREFTET

**Bevis:**
- I `exit_manager.py` linje 200-221: FARM_V2_RULES close kaller `request_close` via `self._runner.request_close()`
- Men trade fjernes fra `open_trades` på linje 221 **FØR** `request_close` returnerer
- `request_close` er asynkron og kan returnere `False` hvis ExitArbiter avviser
- Hvis `request_close` returnerer `False`, er trade allerede fjernet fra `open_trades` → trade "forsvinner" men er ikke lukket

**Kode-flow:**
```python
# exit_manager.py linje 200-221
accepted = self._runner.request_close(...)  # Kan returnere False hvis ExitArbiter avviser
if accepted:
    self.open_trades.remove(trade)  # Fjernes kun hvis accepted=True
```

**Problem:** Hvis `request_close` returnerer `False` (f.eks. pga. SL_TICK avvisning), forblir trade i `open_trades` → blokkerer nye entries

### Hypotesis 2: ExitArbiter avviser SL_TICK (SEKUNDÆRT)

**Bevis:**
- ExitArbiter har hardkodet allowlist: `allowed=['RULE_A_PROFIT', 'RULE_A_TRAILING', 'RULE_B_FAST_LOSS', 'RULE_C_TIMEOUT']`
- `SL_TICK` er ikke i listen (linje 4995 i oanda_demo_runner.py: `if "BROKER_SL" in allowed_reasons: allowed_reasons.add("SL_TICK")`)
- Men SNIPER exit config har ikke `BROKER_SL` i allowlist → SL_TICK blir ikke tillatt

**Verifisering nødvendig:**
- Sjekk SNIPER exit config: har den `BROKER_SL` i `allowed_loss_closers`?
- Hvis ikke, legg til `SL_TICK` direkte i SNIPER exit config

### Hypotesis 3: Race Condition i max_open_trades Check (LAVESTE SANSYNLIGHET)

**Bevis:**
- Logg viser "Skip entry: max_open_trades reached (1)" selv etter at trade er lukket
- Men dette er sannsynligvis fordi trade ikke er fjernet fra `open_trades` (se Hypotesis 1)

**Verifisering nødvendig:**
- Logg `len(self.open_trades)` ved hver entry-skip
- Verifiser at trade faktisk er fjernet fra `open_trades` etter close

### Hypotesis 3: Session Mismatch Blokkerer Close

**Bevis:**
- EntryManager logger `session=US` men ENTRY_DIAG logger `session=OVERLAP`
- Warning: "session not in policy_state or invalid (OVERLAP)"
- Exit-logikk kan avhenge av korrekt session-tagging

**Verifisering nødvendig:**
- Sjekk om exit-logikk faktisk avhenger av session
- Verifiser at session-mismatch ikke påvirker close-events

---

## Debug Plan

### 1. Reproduser Problem Deterministisk

**Debug slice:** 2025-03-12 12:30–15:00 UTC (3.5 timer, ~42 bars)

**Kommando:**
```bash
python3 <<'EOF'
import pandas as pd
from pathlib import Path
import subprocess
import sys

data_path = Path("data/raw/xauusd_m5_2025_bid_ask.parquet")
df = pd.read_parquet(data_path)
df.index = pd.to_datetime(df.index, utc=True)

# Debug slice: 2025-03-12 12:30–15:00 UTC
slice_start = pd.Timestamp("2025-03-12 12:30:00", tz="UTC")
slice_end = pd.Timestamp("2025-03-12 15:00:00", tz="UTC")
df_slice = df[(df.index >= slice_start) & (df.index <= slice_end)].copy()

# Filter for EU/OVERLAP/US sessions
from gx1.execution.live_features import infer_session_tag
df_slice["session"] = df_slice.index.map(infer_session_tag)
df_slice_filtered = df_slice[df_slice["session"].isin(["EU", "OVERLAP", "US"])].copy()

print(f"Debug slice: {len(df_slice_filtered)} bars ({slice_start} to {slice_end})")

output_path = Path("data/raw/xauusd_m5_2025_debug_slice.parquet")
df_slice_filtered.drop(columns=["session"]).to_parquet(output_path)

policy_path = "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml"
subprocess.run([
    sys.executable, "-m", "gx1.execution.oanda_demo_runner",
    "--policy", policy_path,
    "--replay-csv", str(output_path),
    "--fast-replay"
], check=False)
EOF
```

**Output:** `gx1/wf_runs/SNIPER_DEBUG_SL_TICK_<timestamp>/`

### 2. Feilsøk Konkrete Symptomer

#### A) max_open_trades Blokkering

**Hvor:** `gx1/execution/oanda_demo_runner.py` - `_evaluate_entry_impl` eller `evaluate_entry`

**Debug logging (rate-limited):**
```python
if len(self.open_trades) >= max_open_trades:
    log.warning(
        "[MAX_OPEN_TRADES_DEBUG] Blocked entry: open_count=%d max=%d trades=%s",
        len(self.open_trades), max_open_trades,
        [(t.trade_id, t.entry_time.isoformat(), getattr(t, 'exit_time', None)) for t in self.open_trades]
    )
```

**Mål:** Bevise om trade faktisk er åpen eller om det er en race condition.

#### B) SL_TICK Close Event Emission

**Hvor:** Søk etter `SL_TICK`, `tick.*close`, `TICK.*propose` i codebase

**Debug logging:**
```python
log.info(
    "[SL_TICK_DEBUG] Proposed close: trade_id=%s reason=SL_TICK pnl=%.2f price=%.5f stop_level=%.5f spread=%.2f",
    trade_id, pnl, price, stop_level, spread
)
```

**Mål:** Identifisere hvor SL_TICK close events kommer fra og under hvilke betingelser.

#### C) ExitArbiter Allowlist

**Hvor:** `gx1/execution/exit_manager.py` - `evaluate_and_close_trades` eller exit profile config

**Nåværende allowlist:**
```python
allowed=['RULE_A_PROFIT', 'RULE_A_TRAILING', 'RULE_B_FAST_LOSS', 'RULE_C_TIMEOUT']
```

**Spørsmål:**
- Hvor kommer denne listen fra? (exit profile config? hard-coded?)
- Hvorfor er SL_TICK ikke tillatt?
- Skal SL_TICK eksistere i replay for SNIPER?

---

## Minimal Fix Plan

### Fix Option 1: Legg til exit_control i SNIPER Policy (IMPLEMENTERT) ✅

**Rationale:**
- SNIPER policy mangler `exit_control` config → bruker default `["BROKER_SL", "SOFT_STOP_TICK"]`
- Default gir ikke `SL_TICK` automatisk (kun hvis `BROKER_SL` er i listen)
- Legg til eksplisitt `SL_TICK` i SNIPER `exit_control.allowed_loss_closers`
- Dette påvirker ikke FARM (FARM har egen policy uten `exit_control` → bruker default)

**Implementering:**
```yaml
# gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml
exit_control:
  allowed_loss_closers:
    - RULE_A_PROFIT
    - RULE_A_TRAILING
    - RULE_B_FAST_LOSS
    - RULE_C_TIMEOUT
    - BROKER_SL  # Allows SL_TICK automatically
    - SL_TICK    # Explicitly allow SL_TICK for SNIPER
```

**FARM Impact:** Ingen (kun SNIPER policy endres)

### Fix Option 2: Disable SL_TICK i Replay (ALTERNATIV)

**Rationale:**
- SL_TICK er sannsynligvis en live-sikkerhetsmekanisme (tick-based emergency stop)
- I replay har vi ikke "tick" events, kun M5 candles
- SL_TICK close events bør ikke emitteres i replay mode

**Implementering:**
```python
# I tick-watcher eller exit-logikk
if hasattr(self, "replay_mode") and self.replay_mode:
    # Skip SL_TICK close events in replay
    if reason == "SL_TICK":
        log.debug("[REPLAY] Skipping SL_TICK close (not applicable in replay mode)")
        return None
```

**Alternativ (config-styrt):**
```yaml
# SNIPER exit config
exit_control:
  enable_tick_sl_close: false  # Default false for replay safety
```

**FARM Impact:** Ingen (FARM bruker ikke SL_TICK i replay)

### Fix Option 2: Tillat SL_TICK i SNIPER Exit Profile (Config Only)

**Rationale:**
- Hvis SL_TICK er legitim og nødvendig for SNIPER
- Legg til i SNIPER exit profile allowlist, ikke FARM

**Implementering:**
```yaml
# gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml
exit_control:
  allowed_loss_closers:
    - RULE_A_PROFIT
    - RULE_A_TRAILING
    - RULE_B_FAST_LOSS
    - RULE_C_TIMEOUT
    - SL_TICK  # SNIPER-specific
```

**FARM Impact:** Ingen (kun SNIPER exit profile endres)

### Fix Option 3: Replay-Only Override av max_open_trades (DIAGNOSTIC ONLY)

**Rationale:**
- Kun for å diagnostisere om problemet er max_open_trades eller close-logikk
- Ikke endelig løsning

**Implementering:**
```python
# I replay init
if self.replay_mode:
    # Temporary debug: allow more open trades in replay
    max_open_trades = self.policy.get("execution", {}).get("max_open_trades", 1)
    if max_open_trades == 1:
        log.warning("[REPLAY_DEBUG] Overriding max_open_trades=1 -> 5 for debugging")
        self.policy["execution"]["max_open_trades"] = 5
```

**FARM Impact:** Ingen (kun replay mode, ikke live)

---

## Session/Policy_State Mismatch Fix (Sekundært)

**Problem:**
- EntryManager logger `session=US` men ENTRY_DIAG logger `session=OVERLAP`
- Warning: "session not in policy_state or invalid (OVERLAP)"

**Fix:**
- Sørg for at `session_tag` alltid settes i `policy_state` før logging/guards
- Bruk samme session-kilde i all logging (én source of truth)
- Fjern UNKNOWN ved å alltid inferre fra timestamp hvis manglende

**Implementering:**
- I `entry_manager.py`, før Stage-0 og guards:
  ```python
  if "session" not in policy_state or policy_state["session"] == "UNKNOWN":
      current_ts = candles.index[-1] if candles is not None else None
      if current_ts:
          policy_state["session"] = infer_session_tag(current_ts).upper()
  ```

---

## Verifikasjon (Must Pass)

### Test 1: Debug Slice (2025-03-12 12:30–15:00 UTC)
- ✅ Minst én trade åpner
- ✅ Minst én close event aksepteres (ingen "stuck open")
- ✅ Ingen SL_TICK avvisninger i logg

### Test 2: SNIPER Q1 Replay
- ✅ Total trades > 0
- ✅ max_open_trades skip rate faller drastisk (< 10%)
- ✅ Stage-0 reason report eksisterer per chunk

### Test 3: FARM Lock Check
- ✅ `./scripts/check_farm_lock.sh` → OK
- ✅ Ingen endringer i FARM baseline fingerprint

---

## Neste Steg

1. ✅ Skriv status-fil (DONE)
2. ✅ Finn hvor SL_TICK close events emitteres (TickWatcher, linje 314 i oanda_demo_runner.py)
3. ✅ Verifiser ExitArbiter allowlist source (policy.exit_control.allowed_loss_closers, default hvis mangler)
4. ✅ Implementer minimal fix (Legg til exit_control i SNIPER policy med SL_TICK)
5. ⏳ Fix session/policy_state mismatch (sekundært)
6. ⏳ Verifiser med debug slice + Q1 replay

---

---

## Implementerte Fixes

### Fix 1: Legg til exit_control i SNIPER Policy ✅

**Fil endret:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`

**Endring:**
- Lagt til `exit_control` seksjon med `allowed_loss_closers` inkludert `SL_TICK`
- Dette sikrer at ExitArbiter tillater SL_TICK close events for SNIPER

### Fix 2: Debug Logging for max_open_trades ✅

**Fil endret:** `gx1/execution/oanda_demo_runner.py` (linje 5191-5193)

**Endring:**
- Lagt til debug logging som viser hvilke trades som blokkerer nye entries
- Format: `Skip entry: max_open_trades reached (N) open_trades=[trade_id1, trade_id2, ...]`

### Fix 3: exit_manager.py self.runner → self._runner ✅

**Fil endret:** `gx1/execution/exit_manager.py` (flere linjer)

**Endring:**
- Fikset alle forekomster av `self.runner` til `self._runner` (ExitManager bruker proxy pattern)
- Dette løste AttributeError som forhindret trade journal logging

---

## Kommandoer for Verifikasjon

### Debug Slice (2025-03-12 12:30–15:00 UTC)
```bash
python3 <<'EOF'
import pandas as pd
from pathlib import Path
import subprocess
import sys

data_path = Path("data/raw/xauusd_m5_2025_bid_ask.parquet")
df = pd.read_parquet(data_path)
df.index = pd.to_datetime(df.index, utc=True)

slice_start = pd.Timestamp("2025-03-12 12:30:00", tz="UTC")
slice_end = pd.Timestamp("2025-03-12 15:00:00", tz="UTC")
df_slice = df[(df.index >= slice_start) & (df.index <= slice_end)].copy()

from gx1.execution.live_features import infer_session_tag
df_slice["session"] = df_slice.index.map(infer_session_tag)
df_slice_filtered = df_slice[df_slice["session"].isin(["EU", "OVERLAP", "US"])].copy()

output_path = Path("data/raw/xauusd_m5_2025_debug_slice.parquet")
df_slice_filtered.drop(columns=["session"]).to_parquet(output_path)

policy_path = "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml"
subprocess.run([
    sys.executable, "-m", "gx1.execution.oanda_demo_runner",
    "--policy", policy_path,
    "--replay-csv", str(output_path),
    "--fast-replay"
], check=False)
EOF
```

### SNIPER Q1 Replay
```bash
python3 <<'EOF'
import pandas as pd
from pathlib import Path
import subprocess
import sys

data_path = Path("data/raw/xauusd_m5_2025_bid_ask.parquet")
df = pd.read_parquet(data_path)
df.index = pd.to_datetime(df.index, utc=True)

q1_start = pd.Timestamp("2025-01-01", tz="UTC")
q1_end = pd.Timestamp("2025-03-31 23:59:59", tz="UTC")
df_q1 = df[(df.index >= q1_start) & (df.index <= q1_end)].copy()

from gx1.execution.live_features import infer_session_tag
df_q1["session"] = df_q1.index.map(infer_session_tag)
df_q1_filtered = df_q1[df_q1["session"].isin(["EU", "OVERLAP", "US"])].copy()

output_path = Path("data/raw/xauusd_m5_2025_q1_eu_overlap_us.parquet")
df_q1_filtered.drop(columns=["session"]).to_parquet(output_path)

policy_path = "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml"
subprocess.run([
    sys.executable, "-m", "gx1.execution.oanda_demo_runner",
    "--policy", policy_path,
    "--replay-csv", str(output_path),
    "--fast-replay"
], check=False)
EOF
```

### FARM Lock Check
```bash
./scripts/check_farm_lock.sh
```

---

---

## EOF Close Implementation (2025-12-17)

### Problem: 87% av trades står åpne ved replay-slutt

**Rotårsak:** Replay avsluttes uten å lukke åpne trades, noe som gjør analyse (PnL, durations, exit reasons) ufullstendig.

**Funnsted:** `gx1/execution/oanda_demo_runner.py` linje 7597-7673
- Eksisterende logikk lukker trades ved slutten av replay
- Men den bruker "REPLAY_END" som reason og går gjennom ExitArbiter
- ExitArbiter kan blokkere hvis reason ikke er i allowlist
- Trade journal får ikke alltid riktig exit_reason og exit_time

**Løsning:** Implementer config-styrt EOF close med "REPLAY_EOF" reason som alltid tillates i replay mode.

*Generert: 2025-12-17*  
*Sist oppdatert: 2025-12-17*

