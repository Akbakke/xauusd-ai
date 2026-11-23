# Aktuell feature- og konsumrevisjon

Verifisert 20.11.2026 i `/home/andre2/src/GX1_CURRENT`, `work/gx1-current`.
Dette dokumentet oppsummerer ferdige kontroller. `NEXT_RUN_POLICY.json` er
arbeidsstatus; `CURRENT_HANDOVER.md` er overleveringen. En teknisk PASS beviser
ikke bedre retning, generalisering eller økonomisk edge.

## Dekning og modellkobling

CURRENT har 1402 Python-filer og 20251 funksjonsdefinisjoner. Hele kilden er
AST-inventert med import-/kallgraf og dynamiske koblingskandidater. Featureeierne
har 396 funksjoner i 24 moduler. Dette er en eksplisitt nevner for gjennomgangen,
ikke påstand om bevist oppførsel for hver historisk funksjon.

| Modellflate | Feltposter |
|---|---:|
| Signal | 318 |
| Kontinuerlig kontekst | 132 |
| Kategorisk kontekst, session_id | 1 |
| M5/M15/H1/H4/D1, 256 hver | 1280 |
| Sum | 1731 |

Alle signal-/kontekst-/MTF-indekser rutes én gang til de åtte spesialistene.
Alle 132 tidsaliaser beholdes; den generiske snapshotbanen bruker øvrige 186
signalfelt. Alle 61 D1/makrofelt beholdes både i signal og kontekst via
`session_regime_encoder`. Den niende kontekstfamilien er ikke en niende
spesialistencoder. Ingen utilsiktet urutet aktiv feature ble funnet. GC/MBP er
bevart separat og inngår ikke i gjeldende TA-profil.

## Uavhengige tall og klokker

I den historiske worker-kvitteringen betyr taggen `LOCAL_GENUINE_OBSERVED` at
den lagrede signalmatrisen er finite. Providerens `observed` beregnes som
`isfinite(signal)` og er ingen uavhengig kilde-/known-maske. Taggen og
originalkvitteringen bevares. Ekte struktur-/undefined-sertifikater og
historisk førstevintage må vurderes separat.

Bevisrot: `/home/andre2/GX1_RUNS/GX1_FULL_PROJECT_AUDIT_20261120_001`.

- `FULL_NATIVE_NUMERIC001`: alle 256 felt på alle fem tidsrammer, samt lokale
  ikke-makrofelt; 1609 råfeltposter. Hele beholdte rader: M5 389579, M15 129916,
  H1 32575, H4 8590, D1 1669. Finite-/NaN-masker, eksplisitte struktur-/undefined-representasjoner,
  diskrete domener og kausale koblinger, alle 27 arrayheaders/bytebindinger og faktisk normaliseringsaritmetikk
  er kontrollert. Fullført exit0 på 5078,675 sekunder.
- `FULL_MACRO_NUMERIC001`: 61 uavhengig beregnede makroformler og erklært
  tilgjengelighet, 122 feltposter/47528638 celler over 389579 lokale rader.
  Samlet er numerikken og de kontrollerte representasjonsreglene for alle
  1731 råfeltposter kvalifisert; dette beviser ikke full kildeobserverbarhet. Hele de eksplisitt bundne
  offentlige kildecontainerne ble hashet, også bytes etter forseglingen;
  etterforseglede observasjoner/tilgjengelighet inngikk ikke i feltberegningen.
  Ingen native TEST-fil ble lest. En komplett historisk vendor-/revisjonsledger
  er fortsatt ikke bevist.
- `FROZEN_TRAIN_STATS003`: uavhengig rekonstruksjon av hele input-TRAIN før
  målmasker, vindusunioner, kvartiler/skala, domener og alias-eierskap. Alle
  1730 skalaer og float64-ekstrema er bitlike. 1729 sentre er bitlike; ett
  navngitt D1-center er numerisk identisk +0/-0, dokumentert av
  `D1_STATS_DIAG001`. Den uendrede lagrede representasjonen brukes til kontroll
  av faktisk transform. Ingen fit-API, parameteroppdatering eller nye
  modellparametre er brukt. Originale mislykkede auditkvitteringer er bevart.

TRAIN har 354550 issues i 2021–2025; utviklings-VAL har 34934 issues i første
halvår 2026. Native TEST fra 01.07.2026 er forseglet. Retningsruten bruker native
M5 med lukket M15/H1/H4/D1, uten M1-data eller M1-metadata.

## Formelkontrakter og observasjon

Feltvis kildeavstemming dekker 19 produsentgrupper, 388 selvstendige regler og
1731 feltforekomster. FIELD_CONTRACTS001 i den aktive 20-stegs bevisroten binder
formler, enheter, parametre, warmup og known/undefined-policy. Alle 388 reglers deklarerte kildekontrakt er avstemt. FIELD_UNIT_AUTHORITY002
lukker det tidligere enhetsspørsmålet for seks realrate-/breakeven-regler ved
å binde den deklarerte Treasury-serien til Federal Reserves eksplisitte
[prosentenhet i H.15](https://www.federalreserve.gov/releases/h15/). Nivå er prosent per år; differanser og breakeven er prosentpoeng.
Det tidligere UNKNOWN-supplementet bevares. MACRO_ORIGIN_VINTAGE_PATH_REVIEW001 viser nå at alle36registrerte Treasury-
råhasher matcher originale DIRECT_HTTPS-kvitteringer; relé alene gir derfor
ingen ukjent opprinnelse. Historiske førstevintager/revisjoner er fortsatt åpne.
Numerisk finitthet er fortsatt ikke bevis for en kjent indikatorverdi eller
historisk tilgjengelighet. Aktiv kvalifikasjonsstatus eies bare av policy.

To eldre inlinekommentarer beskriver ikke faktisk kode korrekt:

- I `entry_candle_primitives_v1.py` er body-direction duration definert som 1
  på første observerte bar. Relasjoner til forrige bar har én NaN-prefiksrad;
  range-relation duration krever et barpar.
- I `trendline_registry_v1.py` bestemmer fitted band re-entry-geometrien, mens
  `TRENDLINE_RETEST_WINDOW_BARS_V1=2*SWING_LOOKBACK+1=7` separat begrenser
  BROKEN-periodens retestmulighet.

FIELD_COMMENT_CORRECTIONS_REVERTED001 binder eksakte symboler/linjer og
originale fil-/AST-hasher. Inlinekommentarene står igjen fordi de komplette
filene inngår i den fryste kildeidentiteten. Denne dokumentasjonen korrigerer
fortolkningen; snapshot, normalisering, beregninger og vekter er uendret.

## Fysisk input- og råkilderegnskap

INPUT_OBSERVATION_ACCEPTANCE001 måler alle 1731 feltforekomster på eksakte
wholeTRAIN-konsumunioner og 86 navngitte verdi-/predikatrelasjoner. Ingen
kontraktbrudd ble målt. 76 fraværsbetingelser er utøvd og 10 er ikke utøvd;
kildeobserverbarhet for uklassifiserte forhold er fortsatt ukjent. Micro-range
har 98 udefinerte nuller og 10719 gyldige presente nuller i kontekstflaten.
Normaliseringsmasken bevarer de foreskrevne hardnullene. Dette endrer ikke
providerens finitthetsmaske til en uavhengig source-known-maske.

FULL_SOURCE_INTEGRITY_ACCEPTANCE001 binder full gjenbruk av eksisterende
rå-MBA-validator: 416 svar, 18 årparts, 13 tallkolonner og tid. Den rekonstruerer
1215514 canonicalrader og det deklarerte ukesfilterets 1410 eksklusjoner.
En siste grensecandle forblir utenfor de 1215513 aktive før-cutoff-radene.
Historisk markedsstenging er ikke bevist av denne reproduksjonen.
TF_CLOCK_CRITERION_RECONCILIATION002 kvalifiserer den uendrede analytiske
UTC22-/H4UTC02-fasen med uavhengig heltallsfase-/OHLC-orakel og faktisk kausalt
TRAIN-konsum. Kriteriet krever ingen NY-brokerdag. Historisk markedskalender
er fortsatt åpen separat; ingen fase eller numerisk funksjon er endret.

OBSERVATION_ENCODING_SCOPE_REVIEW002 presiserer krav07: en avledet indikator
trenger ingen egen vendorfeed. `observed=isfinite(signal)` er en deklarert
numerisk maske; denne definisjonen alene påviser ingen beregningsfeil.
Kildeekte tilgjengelighet og faktisk udefinert/sensurert semantikk må fortsatt
klassifiseres. Tolv level-recurrencefelt kan ha0 både ved manglende tidligere
anker og reell re-anchor på samme pris. Seks squeeze-release-agefelt er
venstresensurerte før første faktiske release; eksisterende autentisert
nabohistorie/carry kan skille dette. Inlinekommentarens `release_event==age0`
krever gyldig tidligere tilstand; første varme rad er ikke en release.
Faktisk konsum av disse tvetydige tilfellene er ikke målt i denne kilde-
gjennomgangen. Ingen ny inputbit eller numerisk rettelse er begrunnet ennå.

## Faktisk kjøring og ubeviste grenser

Den gjenbrukte ferske initialbaselinen kjørte én modell og 6086 forwards over
hele TRAIN/utviklings-VAL, uten optimizersteg. Initiallogits var lik ny TRAIN-
prior. Dette beviser kjøring gjennom profilen; nullstilte sluttvekter gjør det
uegnet som per-felt sensitivitetsbevis. Den kvalifiserte mekanikkprøven beviser
kald bitlik gjenopptakelse, forventet auxiliary detach og positiv samlet M5-
backbone-gradient etter første head-oppdatering. Den beviser ikke nytte eller
gradient fra hvert felt. Originalkvitteringen beholder sin fryste kildeidentitet.
CURRENT har senere avgrensede vakt-/gjenopptakelsesutvidelser; helfilhashene
er derfor ikke generelt identiske med originalMAIN. Makronøkkelkvalifikasjonen
endrer ingen eksisterende modell-, taps-, input- eller vaktkode.

Det opprinnelige konsumkartet finnes i `CURRENT_FEATURE_CONSUMPTION_CLAIMS001.json`;
dets tidligere auditstatus er historisk. De tre COMPLETE-kvitteringene ovenfor
gir dagens numeriske kvalifikasjon. `CURRENT_CODE_FINAL_STATIC_INDEX001.json`
og sju komplette modulutskiftninger gir dagens kildeinventar:311888 AST-kall
og19693importer, inkludert2538definisjonsheaderkall.
komprimerte funksjons-/avhengighetsregistre finnes via
`CODE_AND_FEATURE_INVENTORY_ARCHIVE.json` under bevisroten. Datert
`FEATURE_AUDIT_INVENTORY_20261017.json` beholdes uendret som historisk kildebevis,
med Git-gjenoppretting for senere pensjonerte kilder. Det er ikke dagens inventar.

Ingen per-felt økonomisk nytte, uavhengig senere generalisering, handelsregel,
live/paper-resultat eller profitt følger av disse kontrollene. Ukjente utfall og
0,3 ATR som nettoendringsproxy beholder sine erklærte begrensninger.
