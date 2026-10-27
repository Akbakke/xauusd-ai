# CURRENT HANDOVER — XAUUSD, retning og bps

Eneste repo: /home/andre2/src/GX1_CURRENT, work/gx1-current, på gx1-3090-lan,
WSL Ubuntu-22.04. Ingen oppdrag fra
andre chatter. Den tidligere lesende retningsrevisjonen er fullført.
Les GX1_RULES helt ved første arbeid; kjør scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY er arbeidsstatus. Originale terminaler overstyrer prosa.

Programrot:
 /home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2

## Aktuell status

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
Åtte spesialistfamilier og D1/makro, kausale avsluttede M5/M15/H1/H4/D1 beholdes.
TRAINmai2024–desember2025; CONTROLjan–mars2026lukket.
Juni2026gjenbrukt utviklings-VAL; TESTfrajuli2026forseglet for tilpasning/utvalg.
Ingen Exit,broker,live/paper,ordre,spending ellerpromotion.
Målet er aktivt og uferdig. Teknisk PASS er ikke dokumentert retningsforbedring.

## Selvstendig retningsmål — candle-fasit og ferskere GC-input

Eget mål 01a17d63-d1bd-7772-bf86-654e3a6db841 følger den godkjente planen i
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DESIGN.json.
Arbeidsstatus eies av NEXT_RUN_POLICY.independent_direction_edge_20261027.
Ingen kontakt eller oppdrag til/fra andre chatter inngår i dette målet.

To rene kontrollfunksjoner og en avgrenset audit-rute er implementert.
Native open sammenlignes med rekonstruert bid/ask-mid først etter bitidentisk
reproduksjon av gammel fasit. Siste-M1-OFI velges på eksakt lukket minutt,
med egne masker og kontraktkontroll; ingen eldre gyldig rad fyller et hull.
Eksisterende mål, modell, trening og feature-registry er ikke endret.
Kilde-/peerreview, AST og diffkontroll er utført. Syntetisk forsøk 001 feilet
før testkjøring; originalkvitteringene er bevart. 002 ble erstattet før kjøring.
003 besto alle128 tester (55prisfasit,27snapshot,46auditgrenser), uten
markedsdata, native-kall eller fit. Kilde-/inputidentitet besto sluttkontroll.
Genuin måling er fullført på118249TRAIN-foreldre fra2024/2025, med bitidentisk
replay av gammel fasit og uendrede kilder/inputs. Native-mid gir56/40/18færre
ikke-nullfortegn ved1/5/15min; ingen LONG/SHORT-fortegn snur. Maksimalt avvik
i5min-avkastning er0.002966bps. Et lite fasitkonsistensavvik er målt;
hovedårsaken til svak retning er ikke dermed påvist. Eksisterende245fasit
beholdes fast i neste A/B-test. En endret fasit krever egen nybaseline.
Brukeren har bekreftet kompaktprofilen som bevisst første scope. Parkerte
fullprofilindikatorer er derfor ikke en feil eller en grunn til utvidelse nå.
Den selvstendige lagrede bps-sammenligningen prioriteres med uendret input.
Alle40 nye syntetiske cases besto. Den genuine sammenligningen er fullført:
initiale retningslogits er bitidentiske på samme256TRAIN-rader; senere17288
foreldre er eksakt joinet til245fasiten. Ingen fit, native-kall eller vekter.
AUC5 går0.503021→0.496823; begge korrigerte dag-/ukeintervaller inkluderer0.
Bps-MSE slår ikke egetTRAIN-gjennomsnitt. Kombinert forhåndsregel erFALSE.
Forsøket er ikke kvalifisert for retningsforbedring eller automatisk utvidelse.
Litt bedre logloss er beskrivende, og er ikke en rang-/edgekvalifisering.
Før native uttrekksutvidelse må det separate returhodets rangering dekkes.
DårligMSE utelukker ikke god rangering. Et eget eksplorativt9-cellers kart
forberedes med samme kompaktinput/foreldre/masker, alle tre hoder og1/5/15min.
Den opprinnelige testen og densFALSE-vedtak bevares bitidentisk; ingen beste-
hodevalg eller promotion følger. Fullprefix0U-arbeid er foreløpig kun metadata.
Ingen ekstra pass, ny genuin enhet eller nye indikatorfelt er admittért nå.
Last-M1-OFI-proposalfiler er kun forberedelse; dagens native profil er uendret.
En kildebevist rekkefølgefeil i proposalens inputhashing er rettet før bruk.
Forberedelsestest001 ga32PASS/5feil: tidligere sealed-path-avvisning og
Numba-cacheforsøk utenfor output. Begge er rettet uten svakere vakter.
Originalkvitteringer er bevart. Nytt forsøk002 besto37/37; samlet205
fokuserte cases er bestått på sine uendrede komponenter. Dette er teknisk
kvalifisering av forberedelsen, med0 genuine OFI-fits og uendret native input.
Neste kilde-design skal avgrense fersk egen0U256-paritet før kun manglende
prefikstredjedel; ingen av disse native enhetene er åpnet ennå.
Ingen CONTROL/TEST, Exit, broker, ordre, spending eller promotion er åpnet.
Målet er fortsatt aktivt; fullprefix-konvergens og treningsutvidelse er uavklart.
Eget9-cellers uttrekk er kildeimplementert og separat kildevurdert. Bare ny
66-case syntetikk er åpnet; lagrede genuine outputs krever ny egen admisjon.
Intervallene gjelder relative AUC-forskjeller, ikke absolutt AUC-over0.5.
Den nye66-case syntetikken besto alle faser med0native/fit/Torch-kall.
Én egen600sek/4G lagret-outputenhet er nå åpnet; alt originalt C-resultat
må gjenskapes eksakt før det nye utforskende resultatområdet publiseres.
Eget9-cellers uttrekk er fullført på17288senere utviklingsforeldre. Alle
gamle C-felter gjenskapt eksakt; original combinedFALSE er bevart. Returhodets
AUC1/5/15 er0.496283/0.496262/0.496172. Alle12korrigerte relative intervaller
inkluderer0, ingen udefinerte resampler. Ved nullterskel er5alltidUP og15
alltidDOWN; dette er retningsandel, ikke dokumentert skilleevne. Ingen ny fit
eller native-kall.231fokuserte cases er nå bestått på sine komponenter.
ALL_HEADS_ROOT_REVIEW_001 og QUALIFICATION_STATE_002 binder bevisene.
Neste avgrensning er fersk egen0U256-paritet og kun manglende prefikstredjedel.
Kompaktinput holdes fast; ingen slik nativeenhet eller ekstra epoch er åpnet.

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
