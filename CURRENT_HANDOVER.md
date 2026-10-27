# CURRENT HANDOVER — XAUUSD, retning og bps

Eneste repo: /home/andre2/src/GX1_CURRENT, work/gx1-current, på gx1-3090-lan,
WSL Ubuntu-22.04. Ingen oppdrag fra
andre chatter. Den tidligere lesende retningsrevisjonen er fullført.
Les GX1_RULES helt ved første arbeid; kjør scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY er arbeidsstatus. Originale terminaler overstyrer prosa.

Programrot:
 /home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2

## Aktuell status

Forskningskontrakt for første gunstige quote-bevegelse er implementert separat
fra eksisterende retning og global MFE. Den skiller same-side prisbevegelse
fra gross markout etter spread og beholder bps-størrelse, M1-rekkefølge-
usikkerhet og hullmasker. Syntetisk engangskontroll er avgrenset åpnet;
den er ikke utført. Faktiske data, modellkobling og trening er ikke autorisert.


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
bekreftet kompaktprofilen som første scope; ingen fullprofil eller nye felt nå.
Første analyse med åtte funn og prioriteringer: `docs/DIRECTION_INDEPENDENT_FIRST_ANALYSIS_20261027.json`.
Egen bevisrot: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63`.

Gjort: candle-fasit118249TRAINforeldre kontrollert mot original245. Nativeopen
versus legacyquote-midpoint ga små null/ikke-null-avvik, ingen fortegnssnuing;
ingen hovedårsak til svak retning dokumentert. Gamle targets beholdes i parene.
Egen lagret bps-sammenligning har256initialbitparitet og17288eksaktesenere
foreldre. AUC5fortegn går0.503021→0.496823; forhåndsregelFALSE. Retur-MSE
slår ikke egetTRAIN-gjennomsnitt. ORIGINALRESULT/kvitteringer er bevart.
Separat utforskende9-cellers head/horizon-kart gjenskaper hele gamle RESULT
kanonisk eksakt. Returhodets AUC1/5/15=.496283/.496262/.496172. Alle12
korrigerte relative intervaller inkluderer0;0udefinerte resampler. Ved
nullterskel er retur5alltidUP og15alltidDOWN; treffrate alene er retningsandel.
0nye model/fit/Torch-kall. ALL_HEADS_ROOT_REVIEW_001 og QUALIFICATION_STATE_002
binder resultatene;231fokuserte cases bestått på respektive komponenter.

Ikke gjort: genuine last-M1-OFI-fits, inpututvidelse, ekstra nativepass eller
fullprefix0U-sammenligning. OFI-proposalen er kun37-case testet forberedelse.
Dagens nativeinput er uendret. Nåværende TRAINprobe256 er ikke hel-TRAIN-
konvergensbevis. Ingen skjult retningsforbedring eller økonomisk edge er påvist.

Neste minste steg: NATIVE_0U_COMPATIBILITY_DESIGN_001 er kun kildeadoptert.
Rett bare konkret gammel metadata-/kildebinding i eksisterendeF-rute, med
strengt opt-in og bevarte legacyregler. Bare fersk egen0U256-paritet og så
manglende presentasjonsdel[31744,47602) kan senere kvalifiseres/admitteres.
Gjenbruk ferdige deler; ikke relanser gamle enheter. Tre metadata-kodefiler og
to eksisterende testfiler er avgrenset. Ingen slik numerisk enhet er åpnet.
Kompaktinput, original276state,267prefixstats, GC, mål og permutasjon beholdes.
TEST/CONTROL, Exit, broker, ordre, spending og promotion er fortsatt stengt.
Målet er aktivt; mer trening og konvergens er uavklart.
Metadata-optin er implementert og separat kildevurdert i fem avgrensede filer.
Syntetikk001 hadde137PASS, men enheten feilet feil forventet136 og avvisning
av egen outputroot ved kvitterings-fsync. Originaler bevart som consumedFAILED.
Ny002 rettet bare antall137 og eksakt owned-root-lesing; kildekode uendret.
002 er COMPLETE med137/137PASS, tre escape-negativer og owned-root-positiv,
0payload/model/fit/update/CUDA.368fokuserte cases er bestått på komponentene.
NATIVE_0U_SYNTHETIC_ROOT_REVIEW_001 og QUALIFICATION_STATE_003 binder dette.
Fersk timing256-recipe forberedes som metadata; genuin admisjon er fortsatt
stengt. Ingen extra epoch, ny indikator eller større nativejobb er åpnet.

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
