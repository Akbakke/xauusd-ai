# CURRENT HANDOVER — XAUUSD, retning og bps

Eneste repo: /home/andre2/src/GX1_CURRENT, work/gx1-current, på gx1-3090-lan,
WSL Ubuntu-22.04. Ingen oppdrag fra
andre chatter. Den tidligere lesende retningsrevisjonen er fullført.
Les GX1_RULES helt ved første arbeid; kjør scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY er arbeidsstatus. Originale terminaler overstyrer prosa.

Programrot:
 /home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2

## Aktuell status

Fast 10min-krysshodeanalyse gjelder fire TRAIN-utviklingsfolder: native5/15-
skårer sammenlignes med ET252 på identiske modne 10min-fasiter. Den første
syntetiske engangskontrollen bestod 8 tilfeller og er konsumert. En senere
kildekontroll fant at ukeblokker på tvers av folder ble resamplet uavhengig,
og at kalenderen ble valgt etter kvalifiseringsmasken. Kilden er rettet til
felles kalendervekter før maskering. Rettelsen bestod 10 syntetiske tilfeller,
og engangsadgangen er konsumert. Første faktiske forsøk stoppet før claim og
payloadlesing da arbeidsområdet ble endret parallelt; forsøket er konsumert.
En ny engangsplan ble fullført på 70 043 sammenlignbare TRAIN-utviklingsutfall.
På 10min-fasit er AUC for 5min-skår native 0,505640 mot ET252 0,515228; begge
99%-intervaller for forskjellen er negative, med ukeintervallet nær null.
For 15min-skår omfatter intervallene null. Skårene er ikke trent for 10min,
og gjenbrukte utviklingsdata dokumenterer ikke handelsedge. Se
docs/DIRECTION_10_CROSS_HEAD_RESULT_20261027.json. Adgangen er konsumert.

Tidligere 5min-referansesammenligning brukte felles 1/5/15/30/60-maske og
utelot 3 182 av 70 294 gyldige 5min-rader. Eksisterende per-horisontresultat
viser native AUC 0,504000 mot ET252 0,512791 på hele 5min-utvalget;
fellesutvalget viser 0,503735 mot 0,513092. Punktforskjellen består på begge;
korrekt felles ukeusikkerhet er fortsatt åpen. Se
docs/DIRECTION_FIVE_MINUTE_COHORT_REVIEW_20261027.json.


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

En avgrenset oppfølging av størrelsesskåren mot eksisterende kausal H1 ATR-z-
fortegn er syntetisk kvalifisert og målt på samme 70376 lagrede foreldre.
Native oppskår AUC for abs(5min-endepunkt)>10bps er 0,551582; H1 ATR-z-fortegn
alene 0,557799, parrede 99%-intervaller for forskjellen inkluderer null. Innen
hver faste H1-tilstand er native størrelses-AUC 0,548139. Fra eksisterende
sign-aggregater er native retnings-AUC innen samme H1-tilstander 0,503991.
Parpopulasjonene er ulike; ingen av differansene er et kausalt bidrag.
Gjenbrukte utviklingsdata og midtprisendepunkt beviser ikke tradingedge.
Se docs/DIRECTION_MAGNITUDE_H1_REGIME_RESULT_20261027.json.
Kildehistorikk i policy skiller nå 19-tilfellers forgjenger fra 22-tilfellers gjeldende kilde; begge faktiske engangsplaner er konsumert.
Se `direction_magnitude_h1_regime_source` i NEXT_RUN_POLICY.json.


En kontroll av eksisterende femminuttsskårer mot større endepunkt er forberedt.
Den bruker fast +/-10 bps og alle modne, observerte foreldre, også nuller.
19 syntetiske tilfeller og én avgrenset måling på seks lagrede TRAIN-payloads
er fullført og konsumert; ingen modell eller rådata ble åpnet. På 70376 modne,
observerte 5min-foreldre er native AUC 0,536890 for opp over 10 bps og 0,440893
for ned over 10 bps med negert oppskår. Dag-/ukeintervallene er nominelle og
gjelder gjenbrukte utviklingsdata. Fra samme aggregerte parsummer er oppskårens
AUC mot stor bevegelse i begge retninger 0,551582, ET252 0,517339; egne intervaller
for størrelsesmålet mangler. R1/ref259 i råresultatet er nullplassholdere, ikke
referansemodeller. Dette peker mot mulig størrelses-/regimeinformasjon, ikke
bevist årsak eller edge. H1 ATR-z fortegnets forskjell i gjennomsnittlig
oppsannsynlighet skifter fortegn mellom folder; dette er kun en grov, etterfølgende
kildekontroll og hverken bekrefter eller utelukker regimeforklaring.
Midtprisendepunkt er ikke første treff eller handel.
Se docs/DIRECTION_FIVE_MINUTE_HURDLE_RESULT_20261027.json. Trening er stengt.
Se `direction_five_minute_hurdle_rank_source` i NEXT_RUN_POLICY.json.


Undersøk årsaken til at det nye bps-tapet ikke forbedret senere5min-rangering,
uten å velge terskel, delgruppe eller horisont etterpå. Bruk først ferdige
aggregerte resultater og eksisterende kausale kontroller. Kandidaten spådde OPP
på 2 024 av 17 197 kvalifiserte5min-rader; logloss var 0,693465 mot 0,692740
for egen prior. Dette peker mot et mulig kalibreringsproblem, men forklarer ikke
svak rangering alene. Videre ny måling krever avgrenset plan og admisjon.
Ingen automatisk flerfoldutvidelse er autorisert. En beregningshorisont er ikke
en maksimal holdetid eller handelsregel.

## Hva som er endret og fortsatt åpent

Forskningsfunksjonen for første gunstige quote er utvidet med en eksplisitt,
fast bps-grense; null beholder opprinnelig fasit. Ugunstig bevegelse måles
fra entryreferansen, ikke fra grensen. 31 syntetiske tilfeller bestod under
4G-vakt; ekte data, modell og trening ble ikke brukt. Kontroll og hash finnes
under `direction_first_favorable_bps_hurdle_source` i NEXT_RUN_POLICY.json.
Grensen gjelder strengt større bevegelse; den er en måledefinisjon, ikke
tapsgrense eller maksimal holdetid. M1-rekkefølge og faktisk nytte er åpne.


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
`NATIVE_0U_RECIPE_METADATA_REVIEW_001.json` binder resultatet. Preflighten var ikke i seg selv en
måling av 0U-paritet og ga ikke kjøreautoritet.

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

Den faste analysen av lagrede utdata er uavhengig reprodusert uten nye
modellkall eller fit. TRAIN5min (47308 kvalifiserte) AUC start/slutt
.507550/.515888; sluttmodellens logloss .693095 er samtidig dårligere enn
fast TRAIN-prevalens .692984 på de samme radene. Alle tre horisonters samlet
eval-BCE er .693073 mot .692944 for samme TRAIN-referanse; dette er en
beskrivende aggregasjon, ikke egen signifikansprøve.

Senere gjenbrukt utvikling5min: 17197 kvalifiserte av 17288, AUC .503021
med korrigerte dag-/ukeintervaller som inkluderer .5. Logloss og Brier er
dårligere enn fast TRAIN-prevalens; 14849/17197 prediksjoner er DOWN ved
den forhåndsdefinerte nullterskelen. Dette er ikke TEST eller økonomisk edge.
Uavhengig review `WHOLE_PREFIX_SIGN_VERIFY_001/RESULT.json` SHA-256
c5df54350f2ef39c08014252f46914f3446e3f2704363f8691180738e966e236;
aggregert diagnose `WHOLE_PREFIX_AGGREGATE_DIAGNOSIS_001.json` SHA-256
0beb6cb9f234e46ed2e2e35d3b4313dbf645ee5b67a3450dc454fdc097361ff2.
På de samme 17197 senere femminuttsradene var gjennomsnittlig oppsannsynlighet
49,504% mot 51,579% observerte opp. Logloss .693514 ligger nær konstanten
.693509 med modellens eget kohortgjennomsnitt; fast TRAIN-referanse .692740.
Dette er en beskrivende etteranalyse, ikke årsaksbevis eller kalibreringsvalg.
En strengt stigende kalibrering endrer ikke skårens AUC på samme rader.
Bevis: `PROBABILITY_LEVEL_DIAGNOSIS_002.json` SHA-256
87f7853fc719bcb9bad987d36f01a155feaa8fad22b1b39550454d379e8e58c6.
Policyens referanse til førsteanalysen er samtidig rettet til aktuell filhash;
tidligere hash pekte på en eldre dokumentversjon, ikke på en modellfil.

Eksisterende GC-M5-bok-OFI var merket observert på 17192 av de samme
17197 kvalifiserte radene; fortegn alene ga AUC .498132. Fortegn av kjent
volumdelta og bok-midrespons ga .491888/.493144. Dette er utforskende
punktverdier uten intervaller: observasjonsmasken beviser ikke full dekning
eller siste-minutt-aktualitet, og måler ikke kombinasjoner eller årsaksbidrag.
Bevis `COMPACT_GC_SIGNS_DIAGNOSIS_001.json` SHA-256
edf74e390f25fde88c0c3de64389746c70c79cdcf29215b4010417ea819a565e.

GC-M5-kildeklokke på 17128 kvalifiserte fold1-foreldre: profil-snapshot-alder
var 0 på alle; spotbasis-alder var høyst ett minutt på 17111, eldre på 13
og manglet på fire. Bokdekning var full på 16977, delvis på 146 og manglet
på fem. Siste M5-OFI-fortegn stemte med lokal input på alle valgte rader.
Profilklokken er snapshottilgjengelighet, ikke siste quote; full bokstøtte
beviser heller ikke siste quote eller executionlatens. Ingen variasjon i
profilalder og bare 13 eldre basisrader betyr at tilført verdi av ferskere
informasjon fortsatt ikke er målt. 69 kvalifiserte rader ble ikke koblet;
grense-radgrupper ble hoppet over. Kun én gjenbrukt fold, ingen modell/fit.
Uavhengig aritmetikk-/kildekontroll, uten ny rådatareplay:
`GC_EXISTING_CONTEXT_CLOCK_REVIEW_001.json`, SHA-256
ae221fc07ce2da847d62907de4fb3501afa1bfd271da3546aba99f473387e4e3.
Engangskjøringen er konsumert; CONTROL, TEST og treningsporter er stengt.

Kompaktmodellens GC-M5-sidebane inneholder allerede et felt for avstand
til siste godkjente, fullt dekkede bokminutt. I lagret pre-CONTROL-kildeutvalg
på 118720 unike M5-rader var 117992 null, 79 positive og 649 ukjent. Dette er
ikke samme populasjon som fold1-prediksjonene, og feltet beskriver ikke siste
faktiske quote eller målt nytte. Bevis `M5_BOOK_TRAILING_SUPPORT_INVENTORY_002.json`
SHA-256 4c385bf3240ce53b8e0410ac9f43d0326aba849aeb04d9dfaede998dbd057b18.

M5-profil-snapshot-alder var nøyaktig null på alle 118249 observerte
2024–2025-TRAIN-beslutningsrader, også i den overlappende fold1-delen.
Dette feltet skiller ikke disse radene og er snapshotklokke, ikke siste quote.
Kompaktprofilen har 2622 aktive signalfelt, inkludert 2410 GC-verdi-/maskefelt;
aliaser skal ikke dobbeltelles som selvstendige indikatorer. Dimensjonen
beviser ikke støy eller årsak til manglende edge. Eget bevis:
`M5_PROFILE_SNAPSHOT_AGE_DIAGNOSIS_002.json` SHA-256
cf6de22b1fe5c4ecaa2864463f90c22d75e690d896fda08b560076652ff62c71;
`COMPACT_INPUT_PROFILE_INVENTORY_001.json` SHA-256
82ae3e43660d0754d85240f9e391653bf4b68202a6856f396c6f2500b16c6c3b.

Lagret 5min-retning er sammenlignet med eksisterende 95min MFE/før-MFE-MAE
Entry-mål i fold1. På 15113 like, observerte LONG/SHORT-foreldre stemte
faktisk framtidig 5min-fortegn med etterpå-bestemt path-side i 10163 tilfeller
(67,25%; majoritetsreferanse 51,82%). Framtidig fortegn er ikke tilgjengelig
for beslutningen. Modellens AUC mot path-side er .508985; korrigerte dag-/
ukeintervaller inkluderer .5. Av 15730 gyldige path-rader er 592 hindsight
FLAT etter eksisterende 0-nytte. Dette er én gjenbrukt fold, ikke handler,
realisert økonomi eller en 95min holding-/Exitregel. 392 ikke-koblede rader
og øvrig modnings-/måldekning er beholdt i kvitteringen. Uavhengig
tidsjoin og punkttall er reprodusert; seks blokkresamples er kontrollert
med fysisk dupliserte rader. Hele 2048-replikaters intervall er ikke
uavhengig kjørt på nytt. Bevis `SIGN_VS_PATH_VERIFIED_ROOT_REVIEW_001.json`
SHA-256 2603916cfdbcffde1d8ba17ff44ef1e8565fa28edff2640c46a2bf9d8db5a980.
Engangskjøringen er konsumert; CONTROL, TEST og trening er stengt.

Den separate 5min-/95min-kontrollen er fullført og konsumert uten modell,
trening, Exit, CONTROL, TEST eller nye indikatorer. Samme gjenbrukte fold
beviser ikke generalisering eller økonomisk edge.

Engangsmålingen av siste M1s kjente kjøps-/salgsregistreringer feilet ved
lagring: tidsvariabelen ble overskrevet med en NumPy-array. Ingen AUC eller
verdiresultat ble lagret. Den ufullstendige kvitteringen viser 2562 strengt
kvalifiserte rader med eksakt fem-M1/M5-paritet, hvorav 2530 også hadde
gyldig 5min-retning, mot 17197 opprinnelige kvalifiserte. Dette er ikke
generaliserbar dekning og ukjent aggressor er ikke gjettet. Original plan
er FEILET OG KONSUMERT; minste rettelse krever ny separat binding. Bevis:
`LAST_M1_TRADE_COUNT_FAILURE_REVIEW_001.json` SHA-256
de75d6834b048a922e7f15e1b7fa4105f293718209ec48608e172aed5bfe2e18.
Ingen modell/trening, nye modellindikatorer, CONTROL, TEST, Exit eller broker.

Separat, tidsvariabelrettet 002 er fullført: på 2530 strengt kvalifiserte
rader ga fast M5-tellingsfortegn AUC .517398 og med siste M1 som forhåndsvalgt
rangeringsforfining .514226; differanse -.003172. Korrigerte parrede dag-/
ukeintervaller inkluderer null. Ingen tilført verdi er dokumentert for denne
ene testen. 2530/17197 opprinnelige rader er bare om lag 15%; ingen general
dom over ukjente/utelatte, M1-OFI, volum eller modellen. Ingen nye
modellindikatorer, fit, CONTROL, TEST, Exit eller broker. 001 er fortsatt
FEILET/KONSUMERT, og 002 er FULLFØRT/KONSUMERT. Uavhengig rådatajoin,
punkttall og alle 4096 fysiske dag-/ukeblokkresamples samsvarer. Dette er
samme fold, ikke en ny framtidsperiode. `LAST_M1_TRADE_COUNT_VERIFIED_ROOT_REVIEW_001.json`
SHA-256 55b14e74936b740988ba6047647bf9d10a69a0ce796e178980d1273ca3cde405.

Den uavhengige M1-/M5-kontrollen er fullført og konsumert uten modell,
fit, nye indikatorer, CONTROL, TEST, Exit eller broker. Ingen bedre
rangering er dokumentert for denne strenge delpopulasjonen.

Utforskende volumgyldig M1-/M5-kontroll er fullført: 16959/17197
retningsegnede rader, AUC .508177 for M5-fortegn og .507971 med siste M1
som fast rangeringsforfining; differanse -.000206. Korrigerte parrede dag-/
ukeintervaller inkluderer null. 14429 rader har ukjente registreringer i M5,
7912 i siste M1, og åtte siste minutter har ingen kjent side. Ukjent er aldri
tilordnet kjøp/salg. Fem M1-tellinger matcher M5, og tidligere 2530-raders
strenge punktresultat er reprodusert. Etteranalyse på samme gjenbrukte fold,
ikke M1-OFI, volumdelta, modellablasjon, ny generalisering eller økonomi.
Uavhengig kontroll med alternativ tidskobling og fysisk gjentakelse av alle
4096 dag-/ukeresamples gjenskaper resultatene på samme data. Kvittering:
`LAST_M1_TRADE_COUNT_VOLUME_VERIFIED_ROOT_REVIEW_001.json` SHA-256
6ee0b7f6266481ad1dbe2834e5fbe678055965f63ccba1f1499de5e2d84ab5ff.
Engangsscope er konsumert; ingen fit, modellendring, nye modellindikatorer,
CONTROL, TEST, Exit eller broker.

Den uavhengige volumgyldige M1-/M5-kontrollen er fullført og konsumert på
samme utforskende fold og populasjon. Den er ikke en ny fremtidig test;
ingen fit, modell, nye indikatorer, CONTROL, TEST, Exit eller broker.

Den originale treningskvitteringen viser 47602 foreldre og 2976 fullførte
bakover-/optimizersteg. Første og siste batch hadde ikke-null gradienter i
alle tre retningshoder og GC-gater/projeksjoner på M5/M15/H1/H4/D1. Det
avklarer kontakt i disse batchene, ikke om treningsmengden var tilstrekkelig
eller om signalene gir retning. Original kvittering: window-0001/RESULT.json
SHA-256 889103d6a81cf725e7ee4e676c0ecf294774dd6eacd7af44b77f9ab871441912.

M1-bok-OFI: Fem kvalifiserte minutter i samme kontrakt gjenskaper M5-OFI
nøyaktig på alle 17066 kvalifiserte barer. På 16977/17197 retningsegnede
rader ga M5-fortegn AUC .498154 og fast siste-M1-forfining .494998;
differanse -.003156, korrigerte dag-/ukeintervaller inkluderer null. Ingen
merverdi er dokumentert for dette fortegnet. Samme gjenbrukte 2025-Q1-fold;
ikke kontinuerlig/ikke-lineær OFI, modellintervensjon eller økonomi.
Uavhengig tidskobling og fysisk gjentakelse av alle 4096 dag-/ukeresamples
stemmer på samme reduserte historiske data, ikke en ny fremtidsperiode.
Kvittering: `LAST_M1_OFI_VERIFIED_ROOT_REVIEW_001.json` SHA-256
421f3a3b91865d8743998844b8ab69690dd9aa79f0cdb667bbda9b2b458840f9.
Engangsscope konsumert; ingen modell, trening, nye indikatorer, CONTROL,
TEST, Exit eller broker.

Den uavhengige OFI-engangskontrollen er fullført og konsumert. Ingen modell,
trening, nye indikatorer, CONTROL, TEST, Exit eller broker ble brukt.

Startmodellens senere prediksjoner finnes ikke. Undersøk først om en nøyaktig
matchet eksponeringssammenligning allerede finnes; ellers forbered separat
plan med treningsportene stengt. Ingen nye indikatorer, treningspass, TEST/
CONTROL, Exit, broker, ordre, spending eller promotion er åpnet. Målet er aktivt.

Et konkret 1-mot-3-pass design står i
`docs/DIRECTION_MATCHED_EXPOSURE_DESIGN_20261027.json`. Det gjenbruker terminal276
og foreslår to pass med samme data, rekkefølge, AdamW og RNG. Eksisterende
énpass-state/cursor kan ikke representere dette; passbevisst tilstand og resume
må kvalifiseres uten å svekke gammel kontrakt. Designet åpner ikke kode,
checkpoint, modell eller trening.

En integrert 1-mot-3-pass-videreføring er implementert i eksisterende state-,
cursor- og treningsløype. Den gjenbruker samme foreldre, rekkefølge, delbatch,
AdamW-historikk og RNG. 353 fokuserte og 154 tilknyttede syntetiske kontroller
bestod på aktuell kilde; opprinnelig terminalstate har identisk tilstandssegl før og etter
baseline-evalueringen. Ingen ekte checkpoint-/data-/modelltilgang eller
trening ble åpnet av denne kilderevisjonen. En ny, separat avgrenset kontroll
må først bevise ekte restore- og prediksjonsparitet under ny kilde, før en
fersk bundet kampanje kan vurderes. 1-mot-3-sammenligningen er ikke utført,
og ingen retningsforbedring er dokumentert. Eksakt status og kildehasher står
i direction_matched_exposure_continuation_source_20261027 i NEXT_RUN_POLICY.json.

Eksisterende frozen-diagnose støtter nå en separat 256-raders terminal-restore
og original prediksjonsparitet med null optimizersteg. En oppdaget manglende
kobling til kvalifisert historisk inputkompatibilitet er rettet også i selve
videreføringen. 353 syntetiske kontroller og metadatakontroll av 57 kildeeiere
bestod. Ingen ekte checkpoint-/data-/modelltilgang eller kjøring er åpnet;
fersk eksakt engangsadmisjon og alle vakter kreves for faktisk paritetstest.
Se direction_continuation_zero_update_source_20261027 i NEXT_RUN_POLICY.json.

Den planlagte fellesgaten for 1 mot 3 pass krever både AUC, logloss og Brier
på dag- og ukeblokker. Feil i én komponent betyr ikke at retningsrangeringen
ikke forbedret seg. Behold fellesgaten, men forhåndsbind og rapporter 5min AUC
og sannsynlighetsskårer separat; senere data må ikke brukes til kalibreringsfit.
Se docs/DIRECTION_EXPOSURE_METRIC_REVIEW_20261027.json.

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
