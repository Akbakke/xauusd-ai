# CURRENT HANDOVER — XAUUSD, retning og bps

Eneste repo: /home/andre2/src/GX1_CURRENT, work/gx1-current, på gx1-3090-lan,
WSL Ubuntu-22.04. Ingen oppdrag fra
andre chatter. Den tidligere lesende retningsrevisjonen er fullført.
Les GX1_RULES helt ved første arbeid; kjør scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY er arbeidsstatus. Originale terminaler overstyrer prosa.

Programrot:
 /home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2

## Aktuell status

Fast 10min-krysshodeanalyse er kildeforberedt for fire TRAIN-utviklingsfolder:
native5/15-skårer sammenlignes med ET252 på identiske modne 10min-fasiter.
En syntetisk engangskontroll er åpnet, men ikke utført; ekte payloadlesing
er ikke åpnet.


10-minuttersutfall kan nå forberedes fra to lagrede femminuttersutfall med
delt prisrad og eksplisitt avrundingsusikkerhet; bps må sammensettes, ikke
adderes direkte. 27 syntetiske kontroller bestod, inkludert prisrad-,
tids-, 15min-paritets- og bpskontroll. Engangsadgangen er konsumert.
Ingen lagrede datapayloads, modellforwards eller trening ble åpnet.


Forskningskontrakt for første gunstige quote-bevegelse er implementert separat
fra eksisterende retning og global MFE. Den skiller same-side prisbevegelse
fra gross markout etter spread og beholder bps-størrelse, M1-rekkefølge-
usikkerhet og hullmasker. Én syntetisk engangskontroll ga 18 passmarkører og
returkode 0. Kontrollprogrammets terminal ble likevel FAILED fordi dobbel -q
fjernet den forventede tekstsummeringen; originallogg og avvik er bevart og
verifisert. Adgangen er konsumert. Ingen faktiske data, modellkobling, trening
eller dokumentert retningsforbedring følger av testen.


DIRECTION_BPS_PREFIX_FIT_FOLD1_20261027_001 er COMPLETE og konsumert.
Fersk signed-bps+BCE-kandidat brukte samme 47 602 TRAIN-foreldre som baseline276:
1/5/15 minutter, B16, seed20260911, samme rekkefølge/AdamW og én gjennomgang.
Originale worker-, progress-, controller- og vaktkvitteringer samsvarer:
2 976 optimizersteg, 4 090 forwards og alle 17 288 senere utviklingsforeldre.
Worker brukte 7 277,822 sekunder. Ett av maksimalt fire vinduer var nødvendig;
ubrukte vinduer2–4 er lukket. Ingen privat kvalifiseringsvekt ble gjenbrukt.
Kilde84086af8 og policy var fryst gjennom hele kjøringen på fysisk boot560.
Modell/eier/cgroup er borte; originallåsen device78/inode35 er ledig.
Windows-task er Disabled. Alle trenings- og launchautorisasjoner er lukket.
BPS-treningsadgangen og denne sammenligningsadgangen er lukket etter konsum.
Originale artefakter, prediksjoner og checkpoints bevares.

Se docs/DIRECTION_BPS_PREFIX_FIT_RESULT_20261027.json og originale
ACTUAL_ROOT_REVIEW.json, VERIFIED_TERMINAL.json og RUNTIME-kvitteringer.
Root har kun kontrollert kilde, JSON, vaktlogg og prosess-/låsmetadata.
Den forhåndsbundne parvise sammenligningen er fullført på 17 288 senere
utviklingsforeldre. Ved5min er kandidatens AUC 0,496823 mot 0,503021 for
baseline276; korrigerte dag-/ukeintervaller for differansen omfatter null.
BPS-MSE forbedres ikke dokumentert mot både prefiksmean og null. Ingen
retningsforbedring er dokumentert. Se docs/DIRECTION_BPS_SAVED_COMPARISON_RESULT_20261027.json.

Forutgående mekanikk er kvalifisert og konsumeres ikke på nytt:
81 syntetiske prefiks-/vaktkontroller, 54-bindingers metadata-preflight og
genuin002 kontroll av delt initialisering, bps-gradient og eksakt resume.
Egen prefiksskala er 2,763416 / 5,256305 / 9,117405 bps ved1/5/15 minutter.
Ingen normalisering ble refittet, og CONTROL/TEST var lukket.

## Neste minste steg

Undersøk årsaken til at det nye bps-tapet ikke forbedret senere5min-rangering,
uten å velge terskel, delgruppe eller horisont etterpå. Bruk først ferdige
aggregerte resultater og eksisterende kausale kontroller. Kandidaten spådde OPP
på 2 024 av 17 197 kvalifiserte5min-rader; logloss var 0,693465 mot 0,692740
for egen prior. Dette peker mot et mulig kalibreringsproblem, men forklarer ikke
svak rangering alene. Videre ny måling krever avgrenset plan og admisjon.
Ingen automatisk flerfoldutvidelse er autorisert. En beregningshorisont er ikke
en maksimal holdetid eller handelsregel.

## Hva som er endret og fortsatt åpent

Opt-in bevarer eksisterende fortegns-BCE og tilfører tre ubundne signed-bps-
middelverdier fra samme udetechede hidden. Loss=BCE+prefiks-RMS-skalert MSE,
fast relativvekt1. Observerte nuller deltar i bps-loss; de lærer ikke FLAT.
Default sign-only-rute beholdes.144syntetiske bps- og68routekontroller bestod.
Størrelse gir ulik supervisjon til+0,1og+50bps; det beviser ikke bedre edge.
Middelverdi gir ikke usikkerhet, første gunstige treff eller ugunstig forløp.
Prisbanens eksisterende pre-MFE-MAE gjelder før første forekomst av horisontens
største gunstige ekstrem, ikke nødvendigvis før første gunstige bevegelse.
Det er en annen kontrakt enn det rene retningsforsøket. Se
docs/DIRECTION_INPUT_AND_FIRST_FAVORABLE_REVIEW_20261027.json.
RMS garanterer ikke balanserte gradienter; store feil kan dominere.

Prisfasiteier støtter minutter1–60, inkludert10, syntetisk kvalifisert med36tester.
Genuin modell og eksisterende targets bruker fortsatt1/5/15.
Ingen ekte utvidet fasit eller10/30/60-/60-outputnative er målt.
Fasit R_h=10000*(mid(D+h)-mid(D))/mid(D), D=M5bar_start+5minutter.
60 marginale utganger er ikke en felles prisbane eller intraminuttrekkefølge.
Senere short krever en ny kausal beslutning. Kontroll mot native open viser
noen null/ikke-null-avvik, men ingen snudde LONG/SHORT-fortegn på målt TRAIN.

## Beholdt årsaksevidens

Småbevegelses-majoritet ble ikke støttet:0<absR<=1bp er39,17/20,20/11,84%
av TRAIN ved1/5/15;26,2694% av første prefiks frossen batchvektet BCE.
SisteB2celle får8ganger B16cellekoeffisienten; skade fra dette er ikke bevist.
Se docs/DIRECTION_BPS_ALLOCATION_RESULT_20261027.json.

Fast lineær slutt-readout forbedret TRAIN-AUC5 fra0,515888til0,539421,
men senere fra0,503021til0,503360.99%dag/ukeintervaller inkluderer null;
senere logloss forverret ved allehorisonter. Kandidathodet innføres ikke.
Se docs/DIRECTION_FROZEN_READOUT_RESULT_20261027.json.
FirefoldnativeAUC5=0,503735motET252=0,513092 på67112samme senereforeldre;
negative korrigerte parvise intervaller.118249TRAINlokale2622felt/masker
hadde eksakt paritet. Full numerisk MTF-paritet er fortsatt åpen.
GC-flow sterk samtidigGC,svak senereXAU i avgrenset analyse.
Endrede vekter, gradienter og lavereTRAINloss beviser ikke nyttig læring.
I C003 falt samlet varians mellom foreldre rett før native276-retningshodet
med ca.98% på tre faste TRAIN-paneler fra initial til slutt. Både faktisk
informasjonstap og årsak er ubevist; konteksten mates også direkte i joint-kilden.
Treningsprefikset var tidsavgrenset, men presentasjonsrekkefølgen var en fast
MT19937-permutasjon, ikke kronologisk. Se docs/DIRECTION_REPRESENTATION_GEOMETRY_REVIEW_20261027.json.
En offline-kontroll av lav-GELU-bidrag er fullført og konsumert. På de samme
256 TRAIN-foreldrene økte den identifiserbare delmengden med lav ideal
GELU-derivert fra 13,9% til 29,0% av koordinatene. På 17 197 senere 5min-rader
var AUC 0,503021 med hele skåren mot 0,502744 uten delmengdens bidrag;
parrede 99% dag-/ukeintervaller for differansen omfatter null. Dette viser
ikke at GELU bør endres og utelukker ikke informasjonstap. Se
docs/DIRECTION_GELU_CONTRIBUTION_RESULT_20261027.json.

M5 Entry: prisforløp minutter–timer, litenMAE før førsteMFE, begge retninger/FLAT.
Fullt prosjekt bevarer åtte spesialistfamilier og D1/makro. Den valgte kompakte
276/277/279/280-profilen aktiverer sju familier: chart_geometry er parkert,
mens 61 D1/makro-felt beholdes. Dette er scope, ikke en avvisning av familien.
TRAINmai2024–desember2025; CONTROLjan–mars2026lukket.
Juni2026gjenbrukt utviklings-VAL; TESTfrajuli2026forseglet for tilpasning/utvalg.
Ingen Exit,broker,live/paper,ordre,spending ellerpromotion.
Målet er aktivt og uferdig. Teknisk PASS er ikke dokumentert retningsforbedring.

## Selvstendig retningsmål — kompakt input og manglende edge

Egen undersøkelse uten kontakt med den andre chatten. Autoritet/status ligger i
`independent_direction_edge_20261027` i NEXT_RUN_POLICY.json. Brukeren har
bekreftet kompaktprofilen som første scope. Førsteanalyse med dokumenterte funn:
`docs/DIRECTION_INDEPENDENT_FIRST_ANALYSIS_20261027.json`. Bevisrot:
`/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63`.

Candle-fasiten for 118249 TRAIN-foreldre har et lite nativeopen/legacy-midpoint-
avvik uten positiv/negativ fortegnssnuing. Sammenligningene beholder original245.
Bps-hovedtesten gir AUC5 .503021→.496823 og forhåndsregel FALSE. I et separat
utforskende ni-cellers kart er returhodets AUC1/5/15 .496283/.496262/.496172;
alle 12 korrigerte relative intervaller inkluderer null. Nullterskel gir alltid
UP på 5 minutter og alltid DOWN på 15; treffrate alene er ikke skilleevne.
Opprinnelig RESULT er gjenskapt eksakt. Ingen økonomisk edge er kvalifisert.

F-kildens strenge metadata-optin er kvalifisert med 137 syntetiske tilfeller;
368 fokuserte tilfeller er bestått totalt på sine respektive komponenter.
001 ble bevart som FAILED på grunn av feil testantall og avvist lesing av egen
outputroot. Ny002 rettet disse to forholdene og besto; ingen payload/model/fit.
En fersk metadata-preflight av `NATIVE_0U_TIMING256_RECIPE_001.json` har bestått
mot eksisterende F-kontrakt og 57 faktiske kildehashes uten payload eller modell.
`NATIVE_0U_RECIPE_METADATA_REVIEW_001.json` binder resultatet. Dette er ikke en
måling av 0U-paritet og åpner ikke native kjøring.

Egen 0U-timing256 er fullført på ny fysisk boot med bestått maskinvakt.
Uavhengig bytekontroll mot originale 256 TRAIN-foreldre besto på alle 16
opprinnelige felt; 16 fremoverkall, null optimizersteg og null reverse.
Kvittering og bevis finnes i `NATIVE_0U_TIMING256_001/ACTUAL_ROOT_REVIEW.json`
(SHA-256 1cd4148e6db49cdbfd792938225126133f2cc2d3c3dd8747c6b2d05808b15d01).
Dette bekrefter målekjeden, ikke læring eller edge. Oppstartsoppgaven er deaktivert.

Egen 0U-prefiksdel [31744,47602) er fullført med 15858 foreldre, 992
fremoverkall, null oppdateringer/reverse og bestått maskinvakt. Uavhengig
kontroll: `NATIVE_0U_PREFIX_PART3_001/ACTUAL_ROOT_REVIEW.json`
(SHA-256 6663b1a04d3e3335c47cab5cdd0ed7b0325a44349b72faf847ea85dbbfef29d8).
Oppstartsoppgaven er deaktivert; ingen edge følger av teknisk PASS.

Den faste analysen av lagrede utdata er fullført uten modellkall eller fit.
Foreløpig TRAIN5min AUC start/slutt .507550/.515888; senere utvikling AUC
.503021 og logloss/Brier dårligere enn fast TRAIN-prevalens. Dette er ikke
ny TEST eller økonomisk edge. `WHOLE_PREFIX_SIGN_COMPARISON_001/RESULT.json`
(SHA-256 25183b69807753897028f8c611c6302b26861c6b3d3972910bba1653d20faa5e).
Egen uavhengig rankdata-replay og full telling av manglende/null/umodne
observasjoner er admittért; fem syntetiske kontroller besto. Startmodellens
senere prediksjoner finnes ikke, så ingen direkte før/etter-test der hevdes.
Ingen nye indikatorer, treningspass, TEST/CONTROL, Exit, broker, ordre,
spending eller promotion er åpnet. Målet er aktivt.

## Bps-hodets separate retningsrangering

Begge tidligere lagrede bps-sammenligninger beregnet retnings-AUC fra
logit_up; mean_return_bps ble vurdert med regresjonsfeil, ikke separat
opp/ned-rangering. Det utelukker ikke at dette andre hodet rangerer retning
annerledes. En uforsøkt, eksplorativ kontroll av de to lagrede OOF-utgangene
er forhåndsspesifisert i
/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_HEAD_RANK_DESIGN_20261027_001.json.
5min er fast primær; 1/15min er sekundære. Samme observasjoner, parrede
dag-/ukeblokker og uendret baseline brukes. Mean-bps er ikke sannsynlighet.
Seks syntetiske kontroller bestod. Engangsanalysen av to lagrede OOF-filer er
fullført og konsumert: AUC5 for bps-hodet 0,496262 mot native276 0,503021
på 17 197 identiske kvalifiserte rader. Parrede 99% dag-/ukeintervaller for
forskjellen omfatter null. Ingen bedre rangering er dokumentert; dette er
gjenbrukt utvikling, uten ny modellkjøring, trening eller økonomisk vurdering.
Ingen ny modellkjøring, trening, terskeltilpasning eller payloadlesing er
autorisert av designet. Resultatene blir utviklingsdiagnostikk, ikke edge.
