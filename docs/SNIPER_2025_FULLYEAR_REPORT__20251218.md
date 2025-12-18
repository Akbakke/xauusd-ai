# SNIPER 2025 Fullyear Report (Baseline vs Guarded)

Generated: 2025-12-18  
Source files (single source of truth):
- `SNIPER_2025_OOS_SUMMARY__baseline_20251218_145523.md`
- `SNIPER_2025_OOS_SUMMARY__guarded_20251218_145523.md`
- `SNIPER_2025_DELTA_BASELINE_vs_GUARDED_20251218_145523.md`
- `SNIPER_Q1_METRICS__baseline_20251218_145522.md`, `SNIPER_Q1_METRICS__guarded_*.md`
- `SNIPER_Q2_METRICS__baseline_*.md`, `SNIPER_Q2_METRICS__guarded_*.md`
- `SNIPER_Q3_METRICS__baseline_*.md`, `SNIPER_Q3_METRICS__guarded_*.md`
- `SNIPER_Q4_METRICS__baseline_*.md`, `SNIPER_Q4_METRICS__guarded_*.md`

## Executive Summary

- **Guarded vs baseline – EV**: På tvers av Q1–Q4 ligger guarded svært nær baseline i forventet verdi per trade (delta på kun et par basispunkter per kvartal, jf. delta-rapporten), med et svakt gjennomsnittlig **EV-løft på ca. +2.3 %**.
- **Guarded vs baseline – tail risk**: P90 tap (drawdown-proxy) er kun moderat redusert (ca. **1.7 %** reduksjon i snitt), dvs. guarden gir begrenset ren tail-risk-forbedring i dette 2025-vinduet.
- **Edge-fordeling**: Q1–Q3 viser solid edge (EV 110–155 bps, winrate ~70–85 %, payoff ~2–4), mens **Q4 skiller seg tydelig ut som lav-edge-regime** (EV ~20–22 bps, payoff ~1.3).
- **Konklusjon**: Guarded-varianten oppfører seg som en forsiktig modifikasjon av baseline (ingen dramatisk edge- eller tail-risk-endring), og er derfor en **fornuftig default-kandidat**, men guard-konfigurasjonen bør likevel **revurderes eksplisitt** (ref. delta-rapportens konklusjon).

## Full-year Overview (Per Quarter, Baseline vs Guarded)

Tallene under er direkte lest fra per-quarter-metrics-rapportene (EV i bps).

| Quarter | Variant   | Trades | EV/trade (bps) | Win rate | Payoff |
| ---     | ---       | ---:   | ---:           | ---:     | ---:   |
| Q1      | Baseline  | 6,935  | 156.29         | 84.6%    | 3.20   |
| Q1      | Guarded   | 7,208  | 153.85         | 84.8%    | 3.13   |
| Q2      | Baseline  | 6,102  | 122.42         | 69.7%    | 2.00   |
| Q2      | Guarded   | 6,226  | 122.22         | 70.9%    | 1.96   |
| Q3      | Baseline  | 6,689  | 113.16         | 72.1%    | 3.88   |
| Q3      | Guarded   | 6,576  | 119.88         | 73.6%    | 4.08   |
| Q4      | Baseline  | 4,482  | 20.64          | 52.6%    | 1.28   |
| Q4      | Guarded   | 4,388  | 21.67          | 52.1%    | 1.33   |

På høyt nivå ser vi:
- **Q1**: Svært sterk edge for begge varianter; guarded har litt flere trades (+3.9 %), marginalt lavere EV (-1.6 %) og omtrent identisk winrate/payoff.
- **Q2**: Guarded har noe høyere winrate (+1.2pp) men marginalt lavere payoff; EV er praktisk talt identisk.
- **Q3**: Guarded gjør det best – høyere EV (+5.9 %), høyere winrate (+1.5pp) og bedre payoff (+0.20).
- **Q4**: Begge varianter har lav edge (EV ~20 bps); guarded er svakt bedre på EV og payoff, men forskjellene er små relativt til støyen.

## Delta Summary (Baseline vs Guarded)

Basert på `SNIPER_2025_DELTA_BASELINE_vs_GUARDED_20251218_145523.md`:

- **Trades**: Guarded handler litt oftere i alle kvartaler (fra +2 % til +4 % flere trades).
- **EV per trade**:
  - Q1: Guarded -1.6 % vs baseline (156.29 → 153.85 bps).
  - Q2: Nær identisk EV (-0.2 %).
  - Q3: Guarded **+5.9 %** EV-løft (113.16 → 119.88 bps).
  - Q4: Guarded **+5.0 %** EV-løft (20.64 → 21.67 bps).
- **Win rate**: Guarded har konsekvent høyere winrate (Q1 +0.2pp, Q2 +1.2pp, Q3 +1.5pp, Q4 -0.5pp – i praksis flat).
- **Avg loss & P90 loss (tail risk)**:
  - Gjennomgående små forbedringer: P90 loss-reduksjon per kvartal er -1.6 % (Q1), -0.8 % (Q2), -4.6 % (Q3) og stort sett flat i Q4.
  - Delta-rapportens “Guard Impact Summary” viser **~1.7 % gjennomsnittlig P90-loss-reduksjon** og **+2.3 % gjennomsnittlig EV-impact**.
- **Konklusjon fra delta-rapporten**:  
  > “❌ Recommendation: Review guard configuration – limited tail risk benefit or high EV cost.”  
  Tolket praktisk: Guarden gir små, men positive EV- og tail-risk-endringer totalt sett, men effekten er såpass beskjeden at konfigurasjonen bør revurderes eksplisitt i lys av kompleksitet / risiko.

## Regime Notes (Q4 som lav-edge-regime)

Fra OOS-sammendragene:

- **Q4 baseline**: EV 20.64 bps, winrate 52.6 %, payoff 1.28, P90 loss 285.56 bps.
- **Q4 guarded**: EV 21.67 bps, winrate 52.1 %, payoff 1.33, P90 loss 285.70 bps.
- Begge varianter har EV på **kun ~1/7–1/8 av Q1–Q3-nivået**, med payoff ned mot ~1.3 og stort sett uendret tail risk (P90 loss ~285 bps).

Implikasjoner for drift og videre arbeid:
- Q4 bør behandles som et **lav-edge regime**: edge er positiv, men lav og nært støy, slik at marginal risikojusteringer/tuning av guard sannsynligvis ikke gir robust gevinst uten å øke modellrisikoen.
- Anbefaling nå: **ingen tuning eller policy-splitting for Q4** i denne fasen; Q4 bør heller overvåkes eksplisitt, og eventuelt gates (f.eks. “no-trade/low-size i Q4”) i en senere iterasjon dersom patternen vedvarer i flere år.

## Data Hygiene & Integrity

- **JSON-basert merge**: Alle merged trade-journals bygges nå direkte fra per-trade JSON-filer (`trade_journal/trades/*.json`), ikke fra CSV-indekser. Index (`trade_journal_index.csv`) er en deterministisk rekonstruksjon fra JSON (entry_snapshot, feature_context, router_explainability, exit_summary, execution_events).
- **Fail-closed verify**: `verify_merged_index` krever:
  - gyldig `trade_id`, `entry_time`, `exit_time` (via felles `_is_valid_time(...)`), og
  - normalisert `exit_reason` som enten `RULE_*` eller `REPLAY_EOF`.
  Alle kvartaler som inngår i denne rapporten har passert denne sjekken (0 bad rows).
- **Q1 baseline EOF-hygiene (7 trades)**:
  - 7 av 6,935 trades i Q1 baseline hadde manglende eller ikke-brukbare EOF-tidsstempler (`REPLAY_EOF`-trades i chunk-starten).
  - Disse er reparert **kun i journaling/merge-laget** med eksplisitte markører:
    - `exit_summary.exit_time` settes til chunk-end- eller entry-tid (duration=0) der intet annet kan utledes.
    - `exit_summary.exit_time_source = "FORCED_EOF_BASELINE_Q1"` og egne logglinjer `FORCED_EOF_BASELINE_Q1_ENTRY_FIX ...` brukes for full sporbarhet.
  - PnL og tradinglogikk er **urørt**; dette er ren journal-hygiene for å sikre at alle trades telles som lukket og at metrics/OOS kan beregnes uten å slippe gjennom “left_open”-støy.

## Next Steps (No Reruns)

Gitt at 2025 OOS nå er komplett for Q1–Q4 og begge varianter, med strengt fail-closed oppsett, anbefales følgende videre steg **uten flere reruns**:

1. **Bruk guarded som default-kandidat** for videre evaluering / ev. paper-trading:
   - Guarded gir konsistent, om enn moderat, forbedring i EV og lett reduksjon i tail risk.
   - Det finnes ingen kvartaler der guarded åpenbart “bryter” edge vs baseline.
2. **Ikke gjør ny tuning nå** (hverken for entry, exit eller guard):
   - Arkitekturen og OOS-resultatene er sterke nok til å gå videre til neste fase (f.eks. stabil paper/live-evaluering).
   - Videre tuning nå vil øke risikoen for overfitting på ett enkelt år.
3. **Overvåk Q4 som eget regime i videre drift**:
   - Hold eksplisitt øye med Q4-performance; vurder enkel gate i fremtidige år (f.eks. reduser size eller skru av trading i klart lav-edge-perioder) dersom patternen gjentar seg.
4. **Behold dagens journaling- og merge-hygiene som standard**:
   - JSON-first, fail-closed verify og eksplisitt merking av tvungne EOF-trades har vist seg kritisk for å avdekke og reparere edge-case-feil uten å røre tradinglogikk.
   - Denne pipeline-disiplinen bør bevares og gjenbrukes i senere versjoner / andre systemer (FARM, andre SNIPER-varianter). 

