# Veien videre — oppdatert 30.09.2026

Gjeldende status eies av [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md). Bruk bare
`/home/andre2/src/GX1_CURRENT`, `work/gx1-current`. Én agent og én tung jobb innen CURRENT.

## Ufullført mål: full B, nå blokkert på kildebevis

Brukeren har bestilt [hele B-løpet og videre kvalifisert modelløp](docs/TA_B_FULL_GOAL_20260930.md).
Målet er ikke oppnådd og er blokkert på manglende kildebevis. Alle seks kilder og faktisk tilgjengelighet
kreves før B-fit. Neste konkrete arbeid er GLD/COT-dekning og komplett kausal
B-implementasjon. Fire ferdige ALFRED-arkiver gjenbrukes.
Makroklokken er implementert; tre fokuserte mekaniske tester besto.
Ekte komponentbygging ble stoppet på 26 VIX-rader der en numerisk verdi er
datert før observasjonsdagen. Ingen tilsvarer siste kjente verdi; samme feilklasse
ga null rader i de tre andre makroseriene. Ingen komponentpanel ble publisert.
Den nye kildegrensen er GLD/COT-dekning og dokumentert VIX-klokke.
VIX-formatkontrollen er fullført: ti av ti celler stemmer, og begge offisielle
formater viser 04.07.2025-observasjonen allerede i 03.07-vintagen.
Dette konkrete avviket er altså i kildeeksporten; klokkerettelse er fortsatt uavklart.
Fullført v37-inputbygging (242 felt) og A-forskningens direkte prisfasit bevares.
Det nye native læringsmålet er ennå ikke innført; se presiseringen i målplanen.
Felles A/B-kjerne er nå implementert: eksakt klokke, samme TRAIN/eval-rader,
bevarte prishorisonter og B-minus-A i felles inferens. Fem fokuserte mekaniske
tester besto på syntetiske data. Kildeimport/fullt panel, ekte registrert B-kjøring
og native mål/trening er fortsatt ufullført. Ingen ny markedsmåling er gjort.
Full B er delvis implementert og umålt; build/trening/evaluering er ikke klare.
Native trening/TEST er fortsatt stengt ved dagens uoppfylte porter.
Ingen løfte om kunstig historikk, lønnsomhet eller et grønt treningsvedtak.

## Fullført neste datasteg for B — 30.09.2026

[Oppfølgingen](docs/TA_B_NEXT_SOURCE_STEP_20260930.md) har hentet og kontrollert
alle fire ALFRED-arkivene: 27 filer og 11 249 valgte historiske versjonsdatoer.
Offentlig nettlesereksport fungerer uten API-nøkkel. Alle fire har bestått
endelig kildeaudit; ingen motstridende duplikater eller overlappende intervaller.
Originalfiler, to avviste metadata-kontroller og den endelige kontrollen er bevart.

Full B er fortsatt umålt. GLD/COT har ikke tilstrekkelig historisk versjonsdekning;
de nye arkivoppslagene ga ni HTTP 429 og én HTTP 503. Nye oppslag er stoppet.
Henteeieren stopper nå resten av en batch etter HTTP 429; fokusert test besto.
Ingen redusert kildevariant, B-fit, TEST, trening, handel eller spending er åpnet.

Dollarindeksens første vintage er 04.02.2019. Ingen tilbakefylling til tidligere
beslutninger. Etter kildekvalifisering må A/B sammenlignes på samme faktiske
TRAIN-/eval-populasjon med publikasjonslag og warmup, etter ny forhåndsregistrering.
Neste datagrense er dokumenterbar GLD-tonnasje og COT 088691 Legacy Futures Only.
Ferdige nedlastinger og kontroller gjenbrukes; ingen jobb kjører og ingen relanseres.

## Historikk: tidligere kildeundersøkelse før nettlesertilgangen ble løst

Brukeren ba 30.09.2026 «Ja undersøk B».
[Undersøkelsen](docs/TA_B_SOURCE_INVESTIGATION_20260930.md) hentet og kontrollerte
ekte historiske GLD- og COT-filer. Full B er fortsatt umålt: to enkeltkopier
dokumenterer ikke sammenhengende publiserings-/versjonsdekning. Den kontrollerte
COT-adressen har bare én arkivkopi 01.03–07.04.2019; GLD-indeksen fikk timeout.
ALFRED-skjemaet virker på Mac, men POST fikk fortsatt timeout etter rettet
submit-felt og 60 s grense. To fokuserte tester besto.

Tre tidligere kildeprøver er avsluttet og hashkontrollert. Makrotilgangen som
manglet i denne historiske undersøkelsen er nå løst i oppfølgingen ovenfor.
Leverandørens generelle vintagefunksjon alene er ikke godkjent GLD/COT-dekning. Ingen B-fit eller redusert
kildevariant er åpnet; A/C-resultatene bevares. Native trening, TEST, handel og
spending forblir stengt.

## Fullført autorisert forberedelse

V37-inputbygging, fersk post-rebuild/readiness, TRAIN/VAL lifecycle-filbindinger,
repo-gjennomgang/testtriage og den bestilte kompleksitetsvurderingen er fullført.
Sluttrapport:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/FINAL_PREPARATION_REVIEW_20260929/FINAL_PREPARATION_REPORT.json`.
Les eksakte bevis via NEXT_RUN_POLICY.json. Alle videreføringer er konsumert.
Bevar opprinnelig RED og ny eksplisitt readiness-recovery; ingen ny rebuild.

- [Kompleksitet og redundans](docs/FEATURE_COMPLEXITY_REVIEW_20260928.md).
- [Repo-dekning, rettelser og teststatus](docs/REPO_REVIEW_20260928.md).
- [Byggets feilrettinger og historikk](docs/NATIVE_PREPARATION_AND_REPO_REVIEW_20260927.md).

Ingen ny fullsuite eller gjentatt bestått datakontroll uten konkret nytt funn.
Alle features, åtte familier og tidsrammer beholdes. Ingen data/checkpoints slettes;
ekstern diskopprydding krever retention-eierens rekkeviddebevis og godkjente plan.

## Bevar de fullførte målingene

[Samlet beslutning 29.09](docs/TA_RESEARCH_DECISION_20260929.md): A/C er
INKONKLUSIV og B er umålt. Oppfølgingen 30.09 gjelder bare kilder. Ingen omkjøring,
ny native kontrakt, utførelsesforskning, modell- eller terskelsøk er åpnet.

## Fullførte historiske spor

- [Konsolidering og tidligere sletting](docs/CONSOLIDATION_20260926.md).
- [Tidlig kalibrering og historisk beslutningskontroll: NO-GO](docs/HISTORY2009W_EARLY_DECISION_RESULT_20260927.md).
- [Modellfrie baselines](docs/MODEL_FREE_BASELINES_RESULT_20260927.md),
  [makrohendelser](docs/MACRO_EVENT_BASELINES_RESULT_20260927.md) og
  [intradag-mekanismer](docs/INTRADAY_MECHANISMS_RESULT_20260927.md).

Disse planene skal ikke relanseres. Målet om kostnadsjustert edge er ikke oppnådd.
