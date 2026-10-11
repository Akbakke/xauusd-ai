# Native D1-kontekst og dag–ukers horisont — designbinding 11.10.2026

Status: **design, ikke lansert.** Brukeren vedtok «Ja, gjør begge» 11.10.2026
kveld. Dette dokumentet binder hva som skal bygges og i hvilken rekkefølge.
Ingen kontrakt, rebuild, recipe eller trening er startet av dokumentet; hvert
steg krever egen binding i NEXT_RUN_POLICY.json og eget budsjett.

## Evidensen som begrunner vedtaket (målt, gjenbrukt utviklingshistorikk)

- 95-minuttersmålet på M5 er lukket negativt: IC 0,01–0,03 under 5,8 bps
  kost; argmax gir FLAT som riktig svar. Flere epoker endrer ikke dette.
- Stage C (forsøk 008/009, ablasjon 011–015, forward 010): et long-timing-
  signal på 20 dagers horisont er treff-familiesignifikant, foldrobust og
  retningsstøttet på 2026H1. Det bæres av kryss-panelet (11 instrumenter) og
  makrodriverne (Treasury 2/10 år nominal og real, VIX, GLD-beholdning), med
  COT-posisjonering som bidrag. Prisstate alene gir ingenting.
- Native transformer har bare XAU-pris på fem tidsrammer. Den mangler akkurat
  de inputs som bærer effekten, og den trenes mot et mål der ingen edge er
  synlig.

## Vedtatt endring A: observert markedsmål på dag–ukers horisont

Eier: `gx1/contracts/entry_observed_market_v1.py` får en eksplisitt bundet
etterfølger (ny SCHEMA_VERSION), i tråd med operatørvedtaket 09.10 om
uavhengig Entry-læring.

- Mål: signert, spread-bevisst netto utfall for LONG og SHORT ved hver
  observasjonshorisont, i bps, samme økonomi-eier som i dag.
- Horisonter: 1, 5 og 20 handelsdager målt på native M5-klokken gjennom
  den kausale D1-sesjonskalenderen (ikke faste barantall; helligdager og
  lukkede sesjoner telles ikke). 20 dager er primær, 5 og 1 er sekundære.
  Ingen horisont er en holdetidsregel.
- Dataset-manifest emitter nøyaktige batchnøkler per horisont
  (regel 17); liveness krever alle.
- Konfidens måles aldri antatt: argmax over Q_L/Q_S/0 per horisont beholdes
  som beslutningssemantikk (regel 3); terskler finnes ikke.

## Vedtatt endring B: ny kontekstfamilie «D1 macro/positioning»

Eier: feltnavn og as-of-join i `gx1/features/d1_macro_context_v1.py`,
tilføyd som `MODEL_NATIVE_CTX_CONT_D1_MACRO_FIELDS` i
`entry_model_native_signal_v1.py` (kontekstskjema v13). Spesialisteier er
`session_regime_encoder` for alle 61 felt (regel 19: nøyaktig én eier): modell-
kontrakten krever at hver spesialist eier sekvens-, MTF- og kontekstpartisjoner,
så en kontekst-bare encoder ville vært ny arkitektur, som vedtaket ikke åpner.
Fordi signalkontrakten eksponerer hvert kontinuerlig kontekstfelt til den lokale
sekvens-/snapshot-stien (regel 13: dimensjoner avledes av eiere), vokser
signalflaten fra 254 til 315 felt (signalskjema v39); de 61 feltene er
trappefunksjoner innen M5-vinduet. Feltene er de samme som forskningsflaten,
med samme tilgjengelighetsregler, lest fra de samme manifestbundne kildene
(pinnet kildespesifikasjon `configs/research/D1_MACRO_CONTEXT_SOURCES_20261011_001.json`):

| Gruppe | Felt | Tilgjengelig ved M5-beslutning |
|---|---|---|
| positioning_cot | cot_noncommercial_net_over_oi, cot_change4reports, cot_open_interest_log_change4reports, cot_commercial_net_over_oi (4 felt) | as-of tirsdag + 7 kalenderdager |
| panel_oanda | 11 instrumenter × logret 1/5/20 d + rv 20 d (44 felt) | siste lukkede D1-sesjonsbar for instrumentet, carry over egne helligdager |
| drivers | 13 felt (real10y nivå/Δ1/Δ5/Δ21, nominal10y, slope10y2y nivå/Δ21, breakeven nivå/Δ21, VIX log nivå/Δ5, GLD log nivå/Δ21) | strengt etter slutt på dag D+1 New York |

- Kalender og dollarindeks er bevisst utelatt (inert / skadelig i ablasjonen).
- Feltene joines as-of til hver M5-beslutningstid, aldri til bar-tid.
- Normalisering tilpasses én gang på hele fysiske TRAIN (regel 18); VAL/TEST
  og serve refitter aldri. Kategoriske domener uendret.
- Åtte eksisterende familier består uendret (regel 4). Dette er en niende
  inputfamilie (datalinje), rutet til sesjons-/regime-spesialisten, ikke en
  erstatning og ikke en ny encoder.
- Materialisering: den berikede M1-/M5-lane-produsenten
  (`build_entry_exit_m1_enriched_frame_v1`) legger feltene på hver rad i
  avgrensede biter som float32; datasett-builderen regner dem på nytt fra de
  pinnede kildene og krever bitparitet (som Group-A), og legger dem selv på
  bare når en splits kilde mangler dem (TEST-konstruksjon under policyens
  eksplisitte leseslutt).
- Train = serve (regel 6): serving-ruten må bygge de samme feltene fra de
  samme as-of-reglene før noen paritetshendelse kan bestås.

## Rekkefølge og budsjetter (hvert steg egen policybinding)

1. Kildekontrakt: manifestbundet D1-input-eier som produserer de 61 feltene
   per M5-beslutningstid fra de hashbundne forskningsresultatene (COT,
   OANDA H1, Treasury, VIX, GLD). CPU audit-klasse. Test: as-of-bevis per
   felt mot forskningspanelet (bit-likhet ved samme beslutningstid).
2. Kontrakter: målets etterfølger (A) og kontekstfamilien (B) med tester,
   signalmanifest-versjon, konstruktormetadata og checkpoint-policy.
3. Datasett-rebuild gjennom eksisterende builder (producer-klasse, egen
   build-run-id, regel 8/15). Rebuild er den tunge jobben; estimat fra forrige
   generasjon står i CURRENT_HANDOVER.md.
4. Fersk nullstegsbaseline på den nye flaten (som FRESH_INITIAL_BASELINE_003).
5. Trening: native campaign, eksakt recipe, 3 seeds, forhåndsregistrert
   stoppregel, CONTROL4096 rank-IC per horisont per epoke. GPU krever fersk
   fysisk Windows-boot per invokasjon og alle eksisterende vakter.
6. Sammenligning mot D1-ridge-journalen på samme beslutningsdager: native må
   slå den frosne 68-feltsflaten på treff og fanget mål før den får autoritet.

## Hva vedtaket ikke åpner

Ingen native launch, TEST-lesing, broker, ordre, spending, promotion eller
ny serving-autoritet. training_enabled=false gjelder til steg 5 er bundet.
