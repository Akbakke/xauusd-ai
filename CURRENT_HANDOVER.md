# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier gjeldende status;
prosess- og terminalkvitteringer overstyrer prosa. Tidligere overleveringer
finnes i Git.

## Gjeldende status

FOLD2_S_FULL_FIT_003_COMPLETE_PRIMARY_REJECTED_EDGE_UNQUALIFIED.
Fold2 S-only full-GC 25-cellers engangsplan er fullført og konsumert.
training_enabled=false, native_training_enabled=false og
full_epoch_training_allowed=false. Full GC er ifølge operatøren pauset
inntil videre (bekreftet 16.11). Windows-oppgaven
GX1RandomAccessCampaignV2 er deaktivert. Ingen relansering, ny modellfit,
TEST, Entry/Exit, broker eller handel er innrømmet.

## Gjort og kontrollert

- Kandidat001 ble forkastet før CLAIM på feil skille mellom featureklokke og
  S-fasitklokke. Privat kandidat002 besto fysisk input-/checkpointkontroll,
  men dens fire optimizersteg var bare mekanikk, ikke full læring.
- Kildepakken 16ad9095, opptaket df978893 og lukket relaunch-gate
  fa875c7d ble pushet. Engangskampanjen kjørte etter fysisk Windows-boot
  655 med 20G/512M/250W-vakter. Den besøkte alle 47 338 modne TRAIN-rader i
  2 959 B16-batcher (siste 10), utførte 2 959 optimizersteg, skrev sju
  checkpoints og skåret samme 4 096 senere utviklingsrader før og etter.
  Terminal exit 0, uendret kilde; signert vakt topp 57 °C kjerne, 66 °C minne,
  138,36 W og 2 292 MiB. TEST ble ikke lest.
- Forhåndsvalgt M5 minutt 5 på 1 392 kjente sider feilet begge referanser:
  FINAL Brier 0,328075 mot fryst prior 0,248755, FINAL AUC 0,471748 mot
  samme-TF nåfortegn 0,499836. Retnings-Brier slo prior i 0/25 celler og
  0/14 senere M5-uker. M5-modellen registrerte 0/347 sertifiserte skifter
  mellom nabopunktene. Den ga gjennomsnittlig 27,4 % UP på rader med 54,5 %
  faktisk UP og valgte DOWN i 20 285/20 480 M5-rad/tidspunktkombinasjoner.
- Ved native endepunkt følger modellens M5- og D1-UP-sannsynligheter
  hverandre nesten helt (korrelasjon 0,999731 over 4 096 senere rader).
  Blant 473 rader med kjent side på begge TF er fasitene ulike i 222,
  mens modellens 0,5-UP/NED-svar er ulike i bare 2. Dette er målt
  TF-diskrimineringssvikt i sluttoutput, ikke bevist årsak i én bestemt port.
  Lagrede vekter viser at de seks TF-/tid-kolonnene i spørringsprojeksjonen
  faktisk ble oppdatert (RMS-endring 0,0310); de var ikke fryst.
- Lagrede checkpoint, lest uten trening på de samme 256 senere radene, viser
  tidlig splitt: ved steg 512 falt M5-aktivitets-Brier 0,280493 → 0,078990,
  mens M5-side-Brier steg 0,243376 → 0,269708. Mellom steg 2560 og 2959
  steg side-Brier 0,271524 → 0,327660, og predikert UP falt 47,2 → 29,8 %.
  De siste 399 stegene bedret side-Brier på en separat prøve av 512 nylige
  TRAIN-rader (0,308953 → 0,265399), men forverret altså senere score.
  Full siste TRAIN-kohort hadde 50,4 % faktisk M5-UP på 2 161 kjente sider;
  enkel M5-fasitubalanse forklarer ikke den senere 29,8 %-prognosen.
  På hele TRAIN er M5/D1-fasiter ulike i 2 072/4 234 felles kjente tilfeller,
  M5/H4 i 3 704/7 470. Samme sluttretning på slike ulike oppgaver er en
  konkret svikt i TF-kondisjonering, ikke bevis mot MTF-data i seg selv.
- Tapet gir M5-side ved minutt 5 én av 25 sidetermene og har i tillegg
  25 aktivitetstermer. I full-GC bruker alle fem tidsrammer felles
  backbone og retningshode; TF-/tidkoordinater legges først til i
  spørringsprojeksjonen. Sluttlogitten varierer mye mer mellom rader enn
  mellom tidsrammer på samme rad. Dette lokaliserer svikten til læringsmål /
  utgangsgrense, men beviser ikke én isolert rotårsak. Eldre TA-kjøringer
  med eget M5-hode og TF-dekodere ga ingen etablert generell M5-gevinst.
- Lesende CPU-portmåling på 256 faste senere rader: lokal åttefamilieport
  la 78,3 % snittvekt på volatilitet/kompresjon etter fit, mot 2,8 % før.
  TF-porten var ulik mellom tidsrammer, men sluttretningene ble nesten like.
  Dette er målt ruting, ikke årsaksbevis for én bestemt indikatorfamilie.
- En separat lesende CPU-replay av initial- og sluttcheckpoint på samme
  1 024 spredte TRAIN-rader fant M5 minutt 5 Brier 0,252211 → 0,291098 og
  AUC 0,517416 → 0,484980 på 366 kjente sider. Dette er en begrenset
  TRAIN-prøve, ikke full TRAIN-score. Alle fem tidsrammers siste retningspunkt
  ble dårligere på denne prøven.
- Samlet bevis og eksakte kvitteringer:
  docs/S_FOLD2_S_ONLY_FULL_FIT_RESULT_REVIEW_20261115.json.
  Tidligere funksjonsrevisjon: docs/FEATURE_AUDIT.md. 25 fokuserte rutetester
  besto før kjøringen; mekanisk PASS er ikke bedre handelsbeslutninger.
- Ikke-GC TA-opptak er gjenbrukt uten modellkjøring: på fast senere
  utvikling fant PASS2 27/293 kjente M5-tilstandsskifter og 31/363
  M5-framtidsskifter mellom prognoser én M5-periode fra hverandre.
  Importert INITIAL fant 52/293 og 58/363 på samme par. Dette er svak
  rullerende skiftesporing, ikke bevis for tidlig varsling eller Edge.
  H4 har 0 tilgjengelige par med eksakt 240 minutters avstand i dette
  utvalget; H4-native skiftekvalitet er ukjent her. Alle fem TF og S/F
  er rapportert i docs/TA_NON_GC_NATIVE_TURN_TRACKING_REVIEW_20261116.json.
- Eldre ikke-GC kontroller viste ingen etablert senere M5-rangeringsgevinst
  fra egne TF-hoder eller lineær lesning før TF-pooling; F M5 femfolds
  utviklings-OOF slo heller ikke fryst TRAIN-prior. Kilderevisjonen fant
  ingen frakoblet genuine TA-feature eller manglende TF, men individuell
  indikatornytte er ubevist. Den målte TA-kjøringen hadde allerede eget
  M5-sidehode og lik-observasjonsnormalisering for M5-side. Nåværende
  TA-tap lærer punktvis aktivitet/side, uten egen framoverskuende skifteterm.
  TA native har bare M5-k=1 (fem minutter) per issue; ingen trenet/
  bevart M5-minuttbane eller direkte skiftsannsynlighet fra samme issue.
  En lesende, framoverskuende kontroll på 556 kjente M5-par sammenligner
  PASS2-prognosen ved issue med observert tilstand nå og fem minutter senere:
  skifte-AUC 0,533695 mot importert INITIAL 0,526155, Brier 0,252937
  mot 0,249471. Parvise dag-/ukeintervaller omfatter både AUC 0,5 og
  null endring; ingen etablert M5-vendingsgevinst. D1 punkt-AUC 0,798
  kommer fra bare 11 dager/5 uker, og parvis gevinst er usikker.
  H4 mangler native par i dette utvalget. ONLINE-paritet for nåtilstanden
  er ikke vist. En tidligere, annen TA-modell ga FLAT som treklasse-argmax
  på alle 4 096 utviklingsrader.
  Dette avgrenser mål-/evalueringsgapet, men isolerer ikke én rotårsak
  eller rettferdiggjør ny modellrunde.
- På et separat, allerede fullført ikke-GC Fold2 TA-retningsløp
  (47 338 TRAIN-rader, M5 direction-only) økte framoverskuende M5-skifte-AUC
  0,457503 → 0,541573 på 531 kjente senere par. Parvise dag-/ukeintervaller
  for gevinsten er positive, og Brier faller 0,264438 → 0,249265; men
  FINAL-AUC-intervallene omfatter 0,5. Dette er avgrenset læringsrespons,
  ikke bevist absolutt diskriminering eller Edge. Det er en annen modell
  og en annen utviklingsperiode enn Fold1 PASS2; årsaksforskjellen er ukjent.
  H4 mangler native par også her. Samme rapport inneholder alle TF.
- Samme lagrede Fold2 M5-par er delt etter kjent nåretning. Innen nå-OPP
  rangerer FINAL framtidig skifte med AUC 0,582090 (285 par, 134 skifter;
  dag-/ukeintervall over 0,5); innen nå-NED er AUC 0,557540 (246 par,
  143 skifter; intervaller omfatter 0,5). Parvis gevinst over INITIAL
  for nå-OPP er usikker per dag, positiv per uke; for nå-NED er den usikker.
  Faktiske skifter er hyppigere fra NED enn OPP (58,1 mot 47,0 %), men
  avledet FINAL-sannsynlighet er i snitt lavere fra NED enn OPP (50,1 mot
  50,9 %). Dag-/ukeintervaller støtter denne omvendte grupperekkefølgen
  på det brukte utviklingsutvalget. Dette er en målt svikt i kondisjonering
  på nåtilstand, ikke bevis for at én indikator eller modellport er årsaken.
  Den bundne modellkontrakten har ikke eksplisitt nå-S som forward-input;
  årsaksverdien av dette er ikke isolert. ONLINE-paritet, uavhengig
  generalisering og Edge er fortsatt ubevist.
- Fold2 TA-direction-only er en komponenttest: den bundne tapsfunksjonen
  optimerte direkte bare M5-side i S/F, ikke aktivitet eller de fire andre
  tidsrammenes sideledd. Modellens egen DOWN/FLAT/UP-mapping gir FLAT på
  alle 4 096 senere rader i hver av de fem TF både før og etter trening.
  Betinget side-AUC er derfor ikke en faktisk fem-TF retningsbeslutning.
  I kontrast ga full-prefix Fold1 PASS2 på M5 481 DOWN, 0 FLAT og 3 615 UP;
  FLAT-kollapsen er ikke universell. Fold2 FINAL sideutganger har fortsatt
  Spearman-korrelasjon minst 0,994 mellom alle TF-par. M5/D1-fasiter
  var ulike i 222/473 felles kjente rader, mens betinget side-svar var
  riktig på begge TF i bare 3 av de 222. Ingen fem-TF kvalifikasjon.

## Ikke gjort / faktisk hindring

Det målte problemet er at boten lærer aktivitet og felles retningsnivå, men
ikke nyttig TF-spesifikk retning eller sertifiserte M5-vendinger i denne
S-kjøringen. Et stort indikatorbibliotek hjelper ikke når M5-sidemålet er
ett av 25 sidemål, fasiten ofte er ukjent, og samme hode presses av alle fem
tidsrammer. Dette er en konkret mål-/utgangskobling; hvilken del som
forårsaker mest skade er ennå ikke isolert. S-fasiten er fem marginale
framtidstilstander per TF, ikke direkte vendetid eller kostet Entry.
Individuell indikatornytte, generalisering og økonomisk Edge er ubevist.

M5 som prognose-/beslutningspunkt er allerede fastlagt i GX1_RULES.md
(18.10/06.11); det er ikke en ny designbeslutning. Brukeren presiserte
16.11 at M5, M15, H1, H4 og D1 hver skal ha egen målt retning/tilstand,
mens øvrige tidsrammer understøtter M5 med kausalt lukket kontekst.
Uenighet mellom tidsrammer er gyldig. Den avsluttede full-GC Fold2-fitten
hadde fem output-plasser, men nesten samme retningssvar på alle TF og
oppfylte ikke funksjonskravet. Full GC er nå pauset; historiske målinger
beholdes uten ny GC-diagnose, kildeendring eller kjøring. Ingen trening,
Entry, TEST eller handel åpnes av denne presiseringen.

## Neste steg

1. Plan003 er konsumert og full-GC S-hypotesen forkastet. Operatøren har
   trukket den uinnrømmede 30-epokers memoriseringsdiagnosen; ingen trening
   åpnes av dette arbeidet.
2. Respekter Full GC-pausen. Bruk eksisterende, allerede målte ikke-GC-
   resultater til å avgrense M5-retning og de fire øvrige TF-retningene uten
   ny modellkjøring. M5 er det etablerte beslutningspunktet; mål om de
   separate TF-prognosene faktisk tilfører M5 verdi før en kildeendring.
   Ingen Full GC-kode, checkpointdiagnose eller ny full-GC-hypotese nå.
3. En senere særskilt innrømmet, kort matchet kontroll må vise både M5-side
   mot fryst referanse og faktisk vendingsdiskriminering før mer trening.
   Senere kronologisk generalisering og alle valgte/åpne posisjoners økonomi
   trengs før Edge. TEST, broker og handel er fortsatt stengt.

Opprydding: historiske overleveringsavsnitt er fjernet her og bevart i Git.
Ingen Python-fil er slettet; gjeldende referanse-, import-, test- og
eierskapsrevisjon beviser ingen frakoblet eier. DATA/RUNS og checkpoints
tilhører retention-eieren.
