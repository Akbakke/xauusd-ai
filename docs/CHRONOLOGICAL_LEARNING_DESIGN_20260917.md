# Ett kronologisk læringsforsøk — design frosset 2026-09-17

**Design og kontroll-ID-er er låst. Forsøket er ikke kjørbart eller autorisert
for trening ennå.** Ingen priser, målverdier eller modellutfall ble lest ved
utvalget; ingen forward, fit, normalisering eller optimizersteg ble kjørt.

| Rolle | Kalender, UTC | Rader |
|---|---|---:|
| TRAIN-kandidater |2025-06-01 inkl. til2026-03-01 ekskl.|49 017|
| Senere utviklingskontroll |2026-03-01 inkl. til2026-06-01 ekskl.|16 278|
| Låst kontroll |Samme senere periode|256|

Utvalget følger eksisterende deterministic_uniform_subsample_indices med
seed20260911/salt1. De256 kontrollene fordeles83/87/86 på mars/april/mai.
Rad-ID-ene og hele kalenderpopulasjonene ligger som hashbundne NPY-filer ved
originalplanen. Ingen utskifting etter at utfall blir kjent. Juni og TEST
inngår ikke. Mars–mai er gjenbrukte utviklingsdata, ikke urørt sluttest.

Fem av kalender-TRAIN-radenes observerte120-overgangers ankerfasiter krysser
2026-03-01. Dette er bare en klokkesjekk av Entry-ankeret; samtlige hjelpetargets
og samplede Exit-tilstander må også kontrolleres før endelig eligibilitet.
Datofilter alene er ikke en fullstendig splitt. Inputhistorikk før juni kan
brukes kausalt, men kontrollperioden skal ikke brukes i normaliseringsfit.

## Hypotese og minste nødvendige rettelse

Status etter designfrys: opt-in-koblingen er implementert og77 syntetiske
kontroller består. Se COHERENT_REFERENCE_ENTRY_20260917.md. DESIGN.json og
utvalgs-/beslutningskriterier er uendret; forsøket er fortsatt ikke kjørbart.
Avsnittene nedenfor beskriver blokkeringen og rettelsen ved designfrys.

Ved designfrys brukte Exit observerte Q_mu-referansemål; Entry brukte gammel,
frossen, rå anker-Q. Collator avviser referansetraces i anchor_views. Det er
en konkret uoverensstemmelse i supervisjonen, ikke grunnlag for ny arkitektur.

Første koderettelse skal være eksplisitt opt-in: samme referansemål ved state0
brukes til Entry via eksisterende first-state- og Entry-eier. Sideverdien er
første gjennomførbare likvidasjon pluss max(gyldig videre Q_mu,0); FLAT er0.
Bevar reward, kostnader, gamma, maskering, ekte terminaler, bootstrap, detach
og provenance. Den eksisterende referansepolicyen119/120 og maksimum120
observerte beregningssteg beholdes; ingen horisontsøk eller holdetidsgrense.
Dette er verdien av en deklarert referansepolicy, ikke optimal Q eller profitt.

Test bare den nødvendige forbindelsen med syntetiske CPU-tilfeller: identiske
Entry/Exit-ankermål, tidsgrenseavvisning, masks/terminal/bootstrap og uendret
legacy-sti. Ingen faktisk datatarget eller modellkjøring er åpnet nå.

## Frosset forsøksramme

- Samme200 features, åtte familier, MTF og modellarkitektur. Alle lærte vekter
  starter ferskt i eksisterende konstruktør. Ingen gammel backbone eller EMA.
- Normalisering av base/context/MTF/lifetime fittes bare på riktig TRAIN-prefix.
  Frisk targetkopi fryses under sammenligningen; ingen target-refresh.
- Ett planlagt budsjett:256 oppdateringer med TRAIN16, maksimalt4096 Entries
  fra eksisterende native rekkefølge. ONLINE ved sluttpunktet velges på forhånd.
  Ingen repetert liten replay, beste-checkpoint-søk eller automatisk forlengelse.
- Før/etter sammenlignes mot samme frosne mål og TRAIN-konstanter. Entry/Exit
  og LONG/SHORT rapporteres separat, med sentrert feil, variasjon og handlinger.
  En konstant forskyvning eller all-FLAT/all-HOLD består ikke porten.
- Senere kontroll bruker fast kalenderuke-bootstrap5000/seed20260911 og
  forhåndsbestemte krav i JSON. Uklart eller negativt resultat åpner ikke mer
  trening. Alle tre måneder rapporteres; ingen kontrolltilpasset justering.
- Økonomivurdering krever først læringsbevis og en separat bundet native-rute
  for samme frosne modell/kohort. Alle valgte handler, kostnader, åpne verdier
  og én-posisjonsreplay skal inngå. Kun eksisterende campaign/vakter,
  VAL256/åtte arbeidere/FP32/TF32av og tre timers VAL-vinduer.

Fersk initialisering, prefix-normaliseringsartefakter, full targeteligibilitet,
4096-raders sampleplan, TRAIN256-probe og eksplisitt TRAIN-kilde/utviklingsrolle
for native evaluator mangler fortsatt. Disse er dokumenterte kjøreblokkeringer,
ikke ekstra modellforsøk. JSON inneholder eksakte eiere og påkrevde bevis.

## Bevis og status

Planoperatøren leser bare seks indeksfelt og TRAIN-M1-klokken. Første forsøk
stoppet fordi first_state_time_ns er barstart, mens targetendepunktet er ved
barstenging. Det ble rettet i leseoperatoren; alle tre ID-filer er eksakt like
før/etter. Ingen produksjonskode ble endret. Begge operatorer er bevart.

Kilde:3d28b56e3eefa77b7bc5d0d182308892aded04f0.
Original DESIGN.json, OPERATOR.py og tre rad-ID-filer:
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_REFERENCE_POLICY_20260916_REFERENCE/FROZEN_READOUT_GENERALIZATION_20260916/CHRONOLOGICAL_LEARNING_DESIGN_20260917_V2

DESIGN SHA256:e073c0e976a348418c34258ef6e35c5483321d126005f2be77e08e2a8d435b36

Speil:handover_snapshot/CHRONOLOGICAL_LEARNING_DESIGN_20260917.json.
