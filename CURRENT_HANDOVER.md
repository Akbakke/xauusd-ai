# CURRENT HANDOVER — XAUUSD, retning og bps

Eneste repo: /home/andre2/src/GX1_CURRENT, work/gx1-current, på gx1-3090-lan,
WSL Ubuntu-22.04. Ingen oppdrag fra
andre chatter. Den tidligere lesende retningsrevisjonen er fullført.
Les GX1_RULES helt ved første arbeid; kjør scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY er arbeidsstatus. Originale terminaler overstyrer prosa.

Programrot:
 /home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2

## Aktuell status

En avgrenset kildetilpasning er admittért for å sammenligne den lagrede ET252-
referansen med GX1 på samme 1 042 førstetreff-foreldre. Ingen ny fit eller
ekte payload er åpnet; referansen har andre inputs og kan ikke gi årsaksbevis.

En naiv partial-set-loss har en bevist degenerert løsning ved 1min/10bps:
alle 17 288 mulige utfallssett inneholder «ingen treff», så en modell som
alltid svarer dette kan oppnå null tapsgrense uten å lære retning. Fasiten er
ikke koblet til trening. Se `docs/DIRECTION_HURDLE_LABEL_IDENTIFIABILITY_20261028.json`.

Matched 5min-sammenligning er COMPLETE og konsumert: Av de samme 1 042
kjente første-treff-retningene stemmer 1 017 med sluttfortegnet og 25 går mot.
Samme skår har AUC 0,426768 mot første treff og 0,431849 mot sluttfortegn på
akkurat disse radene. Dermed forklarer ikke forskjellen mellom de to fasitene
alene svak rangering i dette selekterte utvalget. Ingen ny modell, CONTROL
eller TEST. Se `docs/DIRECTION_FIRST_HURDLE_ENDPOINT_ALIGNMENT_RESULT_20261028.json`.

Første-treff-kontrollen er COMPLETE og konsumert på 17 288 gjenbrukte
fold1-utviklingsforeldre, alle minutter 1–60 og fast 10bps gross quote. Ved
5min nådde 1 042 kjent førstegangsside, 16 230 ingen side, 14 hadde ukjent
rekkefølge og 2 manglet bane. Eksisterende 5min-skår hadde AUC 0,426768 på
de 1 042 selekterte sidene; dag-/ukeintervallene ligger under 0,5. Dette er
ikke all-foreldre-ytelse eller grunnlag for å snu signalet, og modellen er ikke
trent på fasiten. Undersøk først om sluttfortegn og første treff faktisk er
enige på samme 1 042 rader. Ingen modell, CONTROL eller TEST ble åpnet. Se
`docs/DIRECTION_FIRST_HURDLE_OOF_RESULT_20261028.json`.

Endepunktsparitet på 118 249 TRAIN-foreldre var allerede verifisert i
`CANDLE_MIDPOINT_ROOT_REVIEW_001.json` under den uavhengige evidensroten.
En ny identisk kontroll ble unødvendig gjentatt og ga samme aggregater; den er
ikke ny evidens om edge. Duplikatkilde er fjernet fra arbeidsbreet og bevart i
Git. Høy/lav har et tidligere kildeidentitetsbevis, men første-treff-fasiten,
modellkoblingen og faktisk læring er ikke kvalifisert. Se
`docs/DIRECTION_TRAIN_YEAR_QUOTE_PARITY_RESULT_20261028.json`.

Fasiten bevarer også et konservativt sett av mulige utfall ved usikker
rekkefølge. 43 syntetiske kontroller bestod og engangsadgangen er konsumert.
Å maskere ukjente treff alene gir utfallsavhengig utvalg, særlig ved 1min.
Ingen tapsfunksjon, ekte første-treff-fasit eller modellkobling er kvalifisert.

Eksisterende M1-leser åpner hele kildeklokken og kan lese en hel radgruppe før
avskjæring ved TRAIN-grensen. Separate årsfiler 2024 og 2025 er funnet via
metadata, men er ikke åpnet eller verifisert. Kontroller eksakt kildeidentitet,
tids-/radbinding og 1/5/15-fasit før ny ekte måling. Se
`docs/DIRECTION_TRAIN_QUOTE_SOURCE_REVIEW_20261028.json`.

Forskningsfasiten for hvilken side som først passerer en fast bps-grense
er implementert i eksisterende første-gunstige-quote-eier. 42 syntetiske
kontroller bestod, inkludert opp/ned, reversering til start og 1–60min.
Ukjent M1-rekkefølge og hull blir ikke fasit; intet treff er ikke en FLAT-
handlingsregel. Engangsadgangen er konsumert. Ekte data, modellkobling,
trening og faktisk prediktiv verdi er ikke målt.

Parret 1-mot-3-pass TRAIN-sammenligning er fullført og konsumert på samme
47 602 foreldre. 5min TRAIN-AUC økte fra 0,515888 til 0,533534, og logloss
falt fra 0,693095 til 0,692069, mot fast TRAIN-prior 0,692984. På senere
fold1 var AUC 0,503021 mot 0,500950; begge korrigerte differanseintervaller
omfatter null og logloss-punktestimatet forverres. Dette er en målt forskjell
mellom TRAIN-fit og senere utvikling, ikke dokumentert handelsedge eller en
entydig årsak. Se `docs/DIRECTION_MATCHED_EXPOSURE_FULLTRAIN_RESULT_20261028.json`.
Alle engangsadganger og treningsporter er lukket; ingen tung jobb pågår.

Parret full-TRAIN-sammenligning er syntetisk kvalifisert med 23 beståtte
adapterkontroller. Den gjenbruker tre énpass-prefikspartisjoner og ett trepass-
uttrekk, binder foreldre/klokker/utfall/masker og måler 1/5/15min AUC, logloss
og Brier mot énpass og fast TRAIN-prior. Dette er beskrivende treningsfit.
Ingen ekte prediksjonspayload eller modell er åpnet gjennom kildeadgangen.
Den separate engangsadgangen er fullført og konsumert, som beskrevet over.

Full TRAIN-måling for 3-pass-terminalen er COMPLETE og konsumert: 47 602
foreldre, 2 976 B16-forwards, eksakt 256-raders paritet, null optimizer- og
baklengssteg og signert vakt PASS. Windows-oppgaven er deaktivert. Foreløpig
5min TRAIN-logloss er 0,692069 etter tre pass mot 0,693095 etter ett; eksakt
radkobling og parret TRAIN-AUC er fullført som beskrevet over. Resultatet gjelder
treningsfit og viser ikke senere generalisering eller handelsedge. Se
`F_FROZEN_EXTRACTION_threepass_terminal_prefix_CAMPAIGN_20261028_001/ACTUAL_ROOT_REVIEW_001.json`
under programroten.

Den eksisterende frosne måleruten er utvidet for 3-pass-terminalens hele
TRAIN-prefiks og syntetisk kvalifisert: 277 tester bestod under CPU-tak, og
57 kildeeiere og faktiske JSON-bindinger er kontrollert uten payloadlesning.
Kildekvalifiseringen åpnet ingen ekte checkpoint, input eller modellkall;
den etterfølgende, separat admittérte målingen er nå fullført som angitt over.

1-mot-3-gjennomgangene er nå parvis sammenlignet på 17 288 senere fold1-
foreldre. Ved 5min er AUC 0,503021 mot 0,500950; både dag- og ukeintervall
for differansen omfatter null. Tre gjennomganger ga høyere logloss/Brier i
punktestimat ved 1/5/15min. Den forhåndsbundne forbedringsgaten bestås ikke.
Dette er én gjenbrukt utviklingsfold, ikke handelsedge eller en dom over 30
passeringer. Parret TRAIN-vurdering på 47 602 foreldre er nå fullført. Engangsadgangen
for sammenligningen er konsumert og alle treningsporter er lukket. Se
`docs/DIRECTION_MATCHED_EXPOSURE_PAIRED_RESULT_20261028.json`.

Den parvise 1-mot-3-gjennomgangskontrollen er syntetisk kvalifisert, med
identiske observasjoner, parrede dag-/ukeblokker og fast 5min-hovedmål for
AUC, logloss og Brier. Fast antall trekk, seed og lineær kvantilmetode er
bundet før adgang til prediksjonene. 11 adapterkontroller bestod etter siste
kildeendring; ingen ekte payload, modellforward eller optimizer ble brukt.
Den etterfølgende CPU-sammenligningen er fullført og konsumert, som beskrevet
ovenfor. Parret TRAIN-sammenligning for 47 602 foreldre er nå fullført.

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

På samme senere utviklingsfold er AUC-punktene for 1/5/15 minutt henholdsvis
.500882/.503021/.507870. Ved opprinnelig nullterskel svarer modellen NED på
90.86%/86.35%/91.51%, mens faktisk OPP-andel er 49.69%/51.58%/52.39%.
Dette er en terskelskjevhet, ikke en påvist årsak. 1/15-minuttstallene har
ikke egne intervaller og folden er gjenbrukt; ingen horisontseleksjon.
`MULTIHORIZON_SIGN_DIAGNOSIS_002.json` SHA-256
fe243fe3fe95a238026f40c30ee572e2aefeea54ca37de66b94d45c0742443a0.

Kontinuerlig M5-OFI mot fast femdobbel vekt på siste M1 er målt på samme
16977 rader: AUC .495971 versus .489661, differanse -.006309. Korrigerte
parrede dag-/ukeintervaller inkluderer null. Vekten ble valgt etter tidligere
fortegnsresultat; utforskende gjenbrukt fold, ingen modellintervensjon eller
økonomi. Resultat: `LAST_M1_OFI_WEIGHT_RESULT_001.json` SHA-256
85364044ea0d70d2c415ef7a5e8b3958741c2ec8cf16e9793e2c3c8f629752b9.
Uavhengig tidskobling og alle 4096 fysisk dupliserte dag-/ukeresamples
stemmer på de samme reduserte historiske dataene. Kvittering:
`LAST_M1_OFI_WEIGHT_VERIFIED_ROOT_REVIEW_001.json` SHA-256
3a871185b5234b38e9f18bf7c3ab2b0ae7117cb859e6c0efff571e0a1e74bbb0.
Kontrollen er konsumert uten modell, trening, nye indikatorer, CONTROL,
TEST, Exit eller broker.

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

Den nye terminal-restorekontrollen er FULLFØRT og KONSUMERT: originalens
256 faste TRAIN-prediksjoner var bitidentiske etter migrering, og modell-,
buffer-, AdamW-, RNG- og modustilstand ble bevart. Den brukte 16 forwards,
null reverse-/optimizersteg og 200,22 sekunder; originale vakt-/terminal-
kvitteringer er verifisert, oppstartsoppgaven er deaktivert og originalene
bevart. Bevis: DIRECTION_TERMINAL_CONTINUATION_ZERO_UPDATE_PROOF_20261027_001.json
SHA-256 743de80de51229ffb8ace855d288e8a66ea115364b9e9070e00b4e9676242cd7.
Dette er mekanisk paritet, ikke retningsforbedring. Den forbrukte planen skal
ikke relanseres. En ny eksakt 1-mot-3-pass-kampanje og alle vakter kreves før
trening; TEST og CONTROL er stengt. Se dynamic_m5_direction_continuation_zero_update
og direction_matched_exposure_continuation_source_20261027 i NEXT_RUN_POLICY.json.

En ny 1-mot-3-oppskrift er forberedt som metadata, med originalt mål og
samme 47 602 foreldre per pass: 5 952 ekstra optimizersteg, totalt 8 928.
Den er ennå ikke fysisk forhåndskontrollert eller admittért for trening.
Recipe SHA-256 61f0911ae432813af809a03134f1e7c7ee9b0ff1f7a05c972b256ac1a034dd45.
Ingen checkpoint eller markedsdata ble åpnet ved denne forberedelsen.

Forhåndskontrollens filvakt bestod tre positive og sju negative syntetiske
kontroller. Den separate CPU-engangskontrollen av 1-mot-3-oppskriften er
FULLFØRT og KONSUMERT: originalens to navngitte checkpoints fikk eksakt hash,
ingen ble deserialisert, og ingen modell, markedsdata eller trening ble åpnet.
En fersk kampanje med kilde-, maskinvare- og tidsvakter kreves før trening.

En konkret kildekontroll fant at videreføringens filvakt ville avvise den
opprinnelige initialtilstanden under checkpoint-/resumeverifikasjon. Den tillater
nå kun lesing av begge eksakt bundne checkpoints og avviser skriving. 354
fokuserte og 154 tilknyttede syntetiske kontroller bestod, inkludert selve
cursor-verifikasjonen under filvakten. Siden kilde er endret må en ny separat
nullstegskontroll bekrefte paritet. Forrige oppskrift og forhåndskontroll er
historiske og skal ikke relanseres. Trening er stengt.

Runnerens faste FINAL_TRAIN-måling omfatter 256 foreldre. Designets fulle
47 602-raders TRAIN-vurdering krever en separat komplett, fast prefiksuttrekking
etter 3-pass-treningen; den kan ikke hevdes fra 256-raderspanelet.

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

Ny separat nullstegskontroll 002 er admittert på reparert kilde: 256 faste
TRAIN-rader, 16 forwards og null optimizersteg. Den tidligere 001 er konsumert.
Trening er fortsatt stengt; resultat krever opprinnelige guard-/terminalkvitteringer.

Nullstegskontroll 002 er FULLFØRT og KONSUMERT på reparert kilde. Alle 256
TRAIN-prediksjoner var bitidentiske; modell, buffere, AdamW, RNG og moduser
ble bevart. 16 forwards, null optimizersteg, vakt PASS, oppstartsoppgave
deaktivert. Bevis DIRECTION_TERMINAL_CONTINUATION_ZERO_UPDATE_PROOF_20261027_002.json
SHA-256 4afa60e4f9cac17b6b69c672491f504538b7daf19a419aeb030da8e1eba018fe.
Dette er teknisk paritet, ikke retningsedge. Fersk 1-mot-3-oppskrift kreves.

Fersk 1-mot-3-oppskrift 002 er bundet til parity-beviset på reparert kilde.
Én CPU-begrenset metadataforhåndskontroll er admittert for de to eksakt
bundne checkpoint-hashene; ingen modell, markedsdata eller trening er åpnet.
Den tidligere beståtte syntetiske filvakten gjenbrukes etter bytekontroll.

Fersk 1-mot-3-oppskrift 002 har bestått én konsumert, CPU-begrenset
metadataforhåndskontroll: to eksakt bundne checkpoint-hasher, null
deserialiseringer, modellkall eller optimizersteg. Terminal SHA-256
651a20a8f3a3c127008dbfec2fef0de54522e4499bda833e71f9828fde09a5be.
Trening er stengt til separat kampanje og fersk avgrenset admisjon.

En konkret scopekontroll fant at kampanjekontrollørens metadataoperasjoner
hasher bundne checkpointfiler utenfor capped audit. En liten valgfri capped
kontrollvei kvalifiseres før videreføring; vanlig installasjonsvei og alle
øvrige vakter beholdes. Ingen modell eller trening er åpnet.

Den valgfrie capped kontrollveien er implementert. 18 Windows-controller-
kontraktstester, PowerShell-syntaks og et faktisk CPU-capped parserstopp
bestod uten modell eller payload. Opprinnelig installasjon er uendret uten
overstyring; den nye veien må bindes eksplisitt ved neste kampanje.

Den ferske 1-mot-3-pass-kampanjen 002 er avgrenset admittert: to ekstra
pass fra original276, maks 5952 nye optimizersteg, samme mål og data, fire
tidsbegrensede vinduer og full TRAIN-vurdering som separat gjenstående steg.
Kampanjen er ikke materialisert eller startet. TEST/CONTROL og all handel er
stengt. Opt-in capped kontrollør må være bundet før den første oppstarten.

1-mot-3-pass-kampanjen 002 er FULLFØRT og KONSUMERT i to vinduer.
5952 ekstra optimizersteg og 17288 senere foreldre er scoret; andre vindu
hadde null optimizersteg. Begge signerte vakter bestod, oppstartsoppgaven er
deaktivert, trening stengt og vindu 3–4 tilbakekalt. Aggregert logloss og Brier
er høyere etter tre pass ved alle tre horisonter, men AUC og parrede intervaller
er ennå ikke beregnet. Full TRAIN-vurdering på 47602 rader gjenstår.
ACTUAL_ROOT_REVIEW_001.json SHA-256 78a123219f19c70ecdc2a2b1a5b6f5110f0563edcb5484e307e77baad1defd1f. Ingen edge er dokumentert.

En ny kildeavgrensning for parret én-/trepass-evaluering er admittert. Den
skal kontrollere identiske foreldre, klokker, utfall og masker samt beregne
5min AUC og logloss/Brier med felles dag-/ukeblokker. Ingen genuine prediksjoner,
modellkall eller trening åpnes før separat faktisk engangsadmisjon.
