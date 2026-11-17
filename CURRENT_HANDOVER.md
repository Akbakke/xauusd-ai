# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier gjeldende status;
prosess- og terminalkvitteringer overstyrer prosa. Tidligere overleveringer
finnes i Git.

## Gjeldende status

TA_M5_COST_PATH_TARGETS_001_COMPLETE_CONSUMED_NO_EDGE. Én integrert ikke-GC
Fold2-fit på 47 338 TRAIN-rader er fullført; fast apr–jun-skår på 17 536
senere utviklingsrader fant ingen kvalifisert native M5-S/F-retning.
Strukturtreff lærte, men ga ingen påvist M5-sideprognose. Den
forhåndsbundne kostkontrollen valgte FLAT på alle rader og tvungen side
tapte i snitt ved alle seks målepunkter. Det nye, eksakte Fold2/Apr–jun-
målarkivet for M5-kost og prisbane er produsert og uavhengig kontrollert;
det har ikke trent en beslutningsmodell. Alle tre engangsplaner er
konsumert. training_enabled=false, native_training_enabled=false og
full_epoch_training_allowed=false. Full GC er pauset; TEST, Entry/Exit,
ONLINE, broker og handel er lukket. Ingen økonomisk Edge eller uavhengig
handelsgeneralisering er bevist. Brukeren fjernet kravet om nytt «ja»
for hver kvalifisert lokal ikke-GC-kjøring. Stabile langkjøringer
kontrolleres omtrent hver time.

Plan001 SHA `4016255217a3c87bdc84715b6e65707090fa09dcaa1e70a3a1b3455bdac84621`
ble CLAIM-et under fysisk boot 656, men feilet i initial TRAIN-strukturskår
før optimizer ble opprettet: 256 initialforwards, null oppdateringer,
ingen checkpoint eller senere score. Den er konsumert og kan ikke relanseres.

Brukeren innrømmet 16.11 én ikke-GC S CPU-lesning på låste opptak.
Syntetisk preflight under 4G/512M besto. Den ene kjøringen avsluttet
med terminal exit 0: 10 faste CPU-lesninger, fire arkivdekoder og null
modell-forwards/optimizersteg. Engangsgaten er konsumert og stengt.
Etterkontrollen er fullført: ingen M5 side- eller vendingsgevinst er kvalifisert.
Ingen native trening, Full GC, TEST, broker eller handel er åpnet.

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
  TA-tap lærer punktvis aktivitet/side og har ikke et eget skifteoutput.
  En skifte-BCE på samme framtidige sidelogit er imidlertid algebraisk
  identisk med eksisterende side-BCE når nå- og framtidsside er kjent; på
  531 beholdte M5-par er differansen eksakt null. En ekstra slik term ville
  bare vekte parene mer, ikke tilføre ny supervisjon.
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
  Dette avgrenser utgangs-/representasjonsgapet, men isolerer ikke én rotårsak
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
- Samme avgrensede Fold2-kontroll på beholdte TRAIN-rader gir 551
  sertifiserte M5-par: faktisk skifteandel 45,3 % fra OPP og 53,1 % fra
  NED, mens FINAL avleder 51,0 og 50,1 %. Modellens omvendte grupperekkefølge
  finnes altså også her, men dag-/ukeintervall for forskjellen i faktisk
  TRAIN-skifteandel omfatter null. Periodeskiftets bidrag er uavklart.
  Treklasse-svaret er FLAT på alle 4 096 TRAIN-rader i alle fem TF også.
  Kilden definerer S som kausal retningsendring ved 0,5 native ATR, ikke
  fortegnet på én isolert femminutters avkastning; høy skifteandel alene
  avdekker derfor ingen målklokkefeil.

- Den særskilt innrømmede ikke-GC S-lesningen på beholdte 4 096 TRAIN /
  4 096 senere rader er konsumert (10 faste lineære CPU-fits, 6,17 s,
  fire arkivdekoder, null native forwards/oppdateringer). Alle fem TF og
  native M5-vendinger er målt. M5 før pooling ga senere side-AUC 0,507714
  og Brier 0,256642, mot native AUC 0,534733/Brier 0,253292 og fryst
  TRAIN-prior Brier 0,250640. Forverringen mot prior har dag-/ukeintervall
  over null. På 556 sertifiserte M5-par var skifte-AUC 0,543363; nivå- og
  parvise gevinstintervaller omfatter 0,5/null. Faktisk treklasse-M5 traff
  710/1 414 kjente sider mot native 749/1 414 og valgte aldri FLAT.
  M15 før pooling slo de svake decoderne relativt i AUC, men slo ikke fryst
  prior i Brier eller sikkert i absolutt AUC. Lineær før-pooling-lesning
  er dermed forkastet som M5-reparasjon; dette avviser ikke mulig ikke-lineær
  indikatorinformasjon. Kompakt aggregat for alle TF og grenser står i
  docs/TA_S_FIVE_TF_FROZEN_REPRESENTATION_RESULT_REVIEW_20261116.json;
  alle parvise TF-konflikter er bevart med hash i kjøringens REVIEW.json.

- Lesende etterkontroll på samme låste senere M5-S-prediksjoner: native AUC
  innen 67 UTC-dager med begge sider er 0,573465 (dag-bootstrap 95 %
  0,545646–0,603124), mot 0,534733 samlet. PREPOOL har 0,522156 innen dag;
  native minus PREPOOL er 0,051309 (0,017882–0,086977). Bare 1,54 % av
  alle OPP/NED-par er innen slike dager, så dette er en avgrenset, etterpå-
  undersøkt lokal rangering, ikke en kvalifisert beslutning eller Edge.
  M5-S ved eksakt beslutningsklokke kan historisk bygges fra lukket M1 og
  tilgjengelig ATR. I opptaket har 1 319 rader både kjent framtidsside og
  eksakt forrige M5-issue; nåsiden er kjent i bare 556, FLAT i 33 og aktiv
  med ukjent side i 730. De 556 kjente parene har 293 faktiske skifter. En lesende, etterpå-
  undersøkt parvis UTC-dag-bootstrap på de samme hashbundne logittene
  (2 000 trekk, seed 20261116) gir native skifte-AUC 0,527193 mot fryst
  TRAIN-prior 0,522671; differanse 95 % [-0,041848, 0,050085]. Native
  skifte-Brier er 0,253117 mot prior 0,249460; differanse 95 %
  [-0,003979, 0,011750]. Begge intervaller omfatter null. Dette gjenskaper
  rapportens punktmål og viser at innen-dag AUC 0,573465 ikke dokumenterer
  bedre skiftevarsel. ONLINE-paritet er ikke vist. Ingen ny fit, native
  forward eller modelloppdatering ble utført.

- Direkte kilde- og outputkontroll skiller to problemer. I den låste
  full-prefix TA-modellen har M5 eget sidehode; M15/H1/H4/D1 deler ett
  sidehode, men får hver sin faktisk tilkoblede TF-koordinat. På 4 096
  senere S-rader ligger 99,6298 % av de fire ikke-M5-logittenes variasjon
  i et felles radeffektledd og bare 0,0795 % i rad×TF-samspill; parvise
  logittkorrelasjoner er minst 0,998846. Fire TF-er svarer derfor nesten
  likt i denne modellen: på M15/H1 er fasit ulik i 431 kjente par, og
  begge treklassesvar er riktige i 4. M5/D1 har 57/208 riktige
  treklassesvar på fasitkonflikter. En tidligere matchet prøve med egne
  ikke-M5-hoder endret outputene, men ga ingen etablert senere M5-gevinst.
  Delt hode er en målt koblingsvei, ikke bevist eneste årsak til svake svar.
  Fold2 M5-only-tap lærte ikke de fire øvrige sidene direkte og kan ikke
  kvalifisere fem retninger. Brukeren har nå bestemt at en neste TA-kandidat
  skal ha eget retningshode for hver av M5/M15/H1/H4/D1, for både S og F.
  Dette finnes allerede som eksplisitt opt-in `dedicated_non_m5_side=True`
  sammen med M5-hodet; gammel checkpoint bruker fremdeles delt ikke-M5-hode.
  En matchet 256-stegs prøve med separate hoder ga blandede endringer og
  ingen etablert M5-gevinst. Kildesporet viser hvorfor hodesplitt alene er
  utilstrekkelig som arkitekturreparasjon: backbone lager fem 128D-vektorer
  etter kryss-TF-attention, men vekter dem sammen og returnerer bare én
  `entry_q_joint_hidden` på 128D. Både S- og F-dekoderen kopierer denne
  samme vektoren til alle fem TF; først deretter legges TF-one-hot og
  horisontfraksjon til. Det er ingen direkte TF-spesifikk vektor i dagens
  retningshoder.
  Det hashbundne, allerede konsumerte TRAIN-/LATER-opptaket har 4 096 rader
  per panel. Blant M15/H1/H4/D1 utgjør rad×TF-samspill 0,7956/1,7264 %
  av variasjonen i vektorene før pooling (TRAIN/LATER), men bare
  0,0878/0,0795 % i S-sidelogittene og 0,0823/0,0718 % i F. Både før
  og etter pooling dominerer felles radeffekt. Kilde:
  market_state_ta_prepool_capture_20261113_001 i policy, kontrollert mot
  RESULT-, TERMINAL- og opptakshasher uten ny forward. Dette lokaliserer
  målt TF-kollaps, men beviser ikke at direkte TF-vektorer ville forbedre
  M5: den konsumerte lineære før-pooling-kontrollen tapte fortsatt mot M5-
  prior. Målingen alene ga ingen fit- eller Edge-admisjon.

- En lesende kontroll av tre allerede konsumerte TRAIN-kvitteringer avgrenser
  M5-målet uten ny kjøring. I fem kronologiske OOF-folder er aktivitet kjent
  på 88 376 M5-spørringer, men side er kjent på bare 29 576 (33,47 %);
  52 931 (59,89 %) er aktiv med ukjent side og 5 869 er FLAT. Ved egne
  native endepunkter er side kjent på 38,93 % for M15, 39,51 % for H1,
  41,83 % for H4 og 44,37 % for D1; over halvparten av aktivitetsradene
  har ukjent side på hver av disse fire også. På 12 129 M5-par med kjent
  side både nå og fem minutter senere skifter 6 237 (51,42 %);
  foldene ligger mellom 50,42 og 53,11 %. Enkel kausal nåside-persistens har
  Brier 0,514222 mot fryst TRAIN-prior 0,249182 på samme par. En egen
  kronologisk referanse basert på EMA50/200 gir Brier 0,249492 mot
  prior 0,249438 på de 29 576 kjente sidene: ingen stabil enkel
  indikatorgevinst.
  Revisjonen av alle 591 682 observerte TRAIN-M1-minutter fant 349 345
  aktive med ukjent side; 308 149 (88,2 %) kommer fra kontraktens bevisste
  gråsone mellom 0,5 ATR for aktivitet og 1,0 ATR for lokal side. Bare fire
  av 198 069 kandidater med sertifisert side ble ukjente i lokal replay.
  Gråsonetallet gjelder hele M1-båndet, ikke eksakt OOF-M5-fordeling.
  Den allerede fullførte fysiske M5-kontrollen måler derimot 62 687 av
  118 192 komplette TRAIN-femminuttersutfall (53,04 %) i den samme
  0,5–1,0-ATR-gråsonen. Andelen er 50,43–55,50 % i hver av de 20
  TRAIN-månedene; grå M5-celler har median absolutt netto 0,296 ATR og
  1,179 noterte spreads. Dette er observert prisbevegelse,
  ikke automatisk bevist side eller økonomisk verdi. Den hashkontrollerte
  kvitteringen er market_state_gray_physical_20261107_001 i policy;
  ingen ny telling eller modellfit ble gjort her.
  Den faktiske tapskontrakten gir bit5 aktivitetsgradient, men eksakt null
  direkte side-logittgradient. M5-side har allerede en kvalifisert variant
  som normaliserer de kjente sidene; svak M5-score kan derfor ikke forklares
  bare som manglende tapsvekt. Flere epoker skaper ingen sidefasit på bit5.
  Eksakte resultater og SHA-256 står under market_state_m1_duration_20261107_001,
  market_state_time_query_references_20261107_001 og
  market_state_ema_side_reference_20261107_001 i NEXT_RUN_POLICY.json.
  Dette er målt svak supervisjonsdekning og ustabil nåside-referanse, ikke
  bevis for at ikke-lineære indikatorer ikke kan ha informasjon. Varighets-
  kvitteringen har median tre observerte minutter for 3 577 fullt avgrensede
  singleton-løp, hvorav 2 794 kortere enn M5-rytmen på fem minutter. De fleste av
  alle 74 662 slike løp er derimot avgrenset av ukjent etikett; medianen
  beviser ikke at virkelige retningsepisoder vanligvis varer tre minutter.
  Den direkte skifteobservasjonen på 6 237/12 129 kjente M5-par står.
  Kjent, lukket M5-OHLC og kjent framtidig side er forskjellige størrelser. Den
  faktiske ikke-GC TA-profilen fører 317 lokale signalfelt og 255 felt per
  lukket TF-vindu til modellen, blant dem bekreftede swing-/pivotnivåer,
  BOS/CHOCH, sweeps, nivåavstand, brudd og retest. Dette er felt, ikke 255
  uavhengige indikatorer eller bevist nytte. SMC bekrefter en pivot først
  etter tre påfølgende lukkede barer; den kan ikke være kjent på pivotbaren.
  Kildetapen har også 95 ukjente kildegap blant 591 682 TRAIN-M1-rader og
  413 dokumenterte lukningsgap, så «alle barer» gjelder bare observerte,
  lukkede barer. På det målte M1-båndet skyldes de fleste bit5 fasitens
  mellomkategori, ikke et hull i observert OHLC. S-tilstanden beregnes
  fra rekkefølgen på lukkede M1-OHLC, mens ikke-GC TA-modellens ni aktive
  featureflater er M5/MTF/kontekst uten M1-bane. Et rent kildeeksempel med
  terskel 1,0 viser dette formelt, med hver M1-H/L lik maksimum/minimum
  av barens åpning og slutt: M1-sluttstiene
  `0, -1.2, -0.8, +1.2, +0.4, 0` og
  `0, +1.2, +0.8, -1.2, -0.4, 0` har lik aggregert M5-OHLC
  `(open=0, high=+1.2, low=-1.2, close=0)`, men `_step_direction` gir
  NED for første og OPP for andre ved M5-slutt. Selv alle korrekte
  M5-barer kan derfor ikke entydig rekonstruere nå-S; faktisk forekomst
  og bidrag til modellfeilen
  er ikke målt. En eventuell M1-tilstand som input krever eksakt lukket
  beslutningsklokke og ONLINE-paritet; eksemplet åpner ingen ny inputrute.
  Den eldre `NativeDynamicMTFStateResearch(current_state_input=True)` har
  riktignok egen nåtilstandsprojeksjon, men kontrakten merker den
  `current_input_available_online=false` og `historical_event_time_only`;
  dens forward krever dessuten fem GC-sekvenser. Den er ingen kvalifisert
  ikke-GC TA-/ONLINE-rute og gjenbrukes ikke blindt mens Full GC er pauset.
  En vendefasit må bruke nivåer kjent ved beslutningen og senere faktisk
  nivåtreff/bekreftelse som utfall, med særskilt håndtering av
  manglende og tvetydige forløp; ingen slik fasit er låst eller fitet nå.

- Lesende Fold2 M5-only-før/etterkontroll på et annet, låst panel enn
  prepool-lesningen: beholdt TRAIN S-side (1 402 kjent) går AUC
  0,489089→0,522454 og Brier 0,259222→0,249832, men fryst prior er
  0,249659. Senere utvikling (1 392 kjent) går AUC 0,521870→0,548964
  og Brier 0,262975→0,249722, fortsatt svakere enn prior 0,248755.
  Etterpå beregnet dag-/uke-bootstrap (2 000, seed 20261116) gir slutt-
  AUC 95 % [0,527809,0,571690]/[0,527510,0,576424] over 0,5,
  men gevinst mot INITIAL har intervaller som omfatter null. Dette viser
  begrenset absolutt S-rangering i gjenbrukt utvikling, ikke bevist
  treningsgevinst, M5-vending, handel eller Edge. Senere S-modell gir
  49,50 % OPP mot 54,53 % faktisk; mot fryst prior gir ekstra kalibrerings-
  skjevhet +0,001719 Brier og sannsynlighetsspredning +0,000093, mens
  prediksjon/utfall-samvariasjon motvirker -0,000845. Netto Brier-gap er
  +0,000968.
  S/F-fasiter er like på 924/955 senere M5-rader med begge sider kjent,
  men dette er bare overlappet: 437 ekstra har kjent S-side og F=FLAT,
  mens 636 har kjent F-side og S=aktiv/side ukjent. Beholdt TRAIN har
  tilsvarende 1 025 begge-kjent, 377 bare S og 636 bare F. Disse er ulike
  målmasker, ikke to like oppgaver som bare mangler felles hode. På de 955
  senere begge-kjente er sluttlogittene likevel negativt korrelert
  (-0,5298). F-side har senere AUC 0,487387/Brier 0,251524 mot prior
  0,249257. Koden kobler begge M5-hoder til tap og optimizer; en tidligere
  bundet felles S/F-hodekontroll ga identiske M5-logitter, men F-AUC
  0,508735/Brier 0,250042, fortsatt ikke kvalifisert. Direkte motsatt
  kjent S/F-side er sjelden (31/955), men ulik maske/tilstandssemantikk
  er målt. Deres bidrag til svak rangering og kalibrering er ikke isolert.
  Kilder: market_state_ta_fold2_m5_direction_fit_20261114_004 i policy og
  docs/TA_FOLD2_SHARED_M5_FIT_RESULT_REVIEW_20261114.json. Ingen ny fit.

- En ny avskrudd-som-standard ikke-GC-forskningsrute
  `direct_tf_hidden=True` bevarer fem post-attention TF-token frem til
  S/F-dekoderen. Den bruker eksisterende entry-Q-projeksjon uten nye
  backbone-parametre og krever egne sidehoder for alle fem TF. Feature view,
  anchor-span, S/F-tap og numerisk oppskrift binder samme eksplisitte flagg;
  gammel standardrute og kontrakthasher er bevart. Fire nye syntetiske
  fokustester kontrollerer TF-lokal dekoding, ett encoder-kall, backbone-
  utgang og view-/oppskriftsbinding. Eldre berørte testpakker er kontrollert.
  Dette er mekanikk og kildekandidat, ikke fysisk native forward,
  læringsgevinst, ONLINE-paritet, fit-admisjon eller Edge.
- Strukturkandidatens dokumentstatus er rettet til faktisk målt Fold2-
  dekning. Den faste SMC-persistensregelen tapte mot fryst TRAIN-prior;
  ingen flere enkeltregelrunder følger.

- Kildesammenligning avdekket at den eldre historiske nåtilstandsinputen
  brukte 0,5 ATR også for side, mens den valgte S-tapen bruker 0,5 ATR for
  aktivitet og 1,0 ATR for sikker side. Eldre accessor kan derfor ikke
  ommerkes som dagens S-nåinput. Den publiserte, hashbundne TRAIN-M1-tapen
  får nå et separat `gather_current`: eksakt lukket M1-tilstand ved bundet
  M5-issue, med bit5 `{OPP,NED}`, sti-/gapmasker og kildeindeks, men uten
  framtidsspørring. Fire nye fokustester består; én bruker tre faktiske
  Fold2-issues, og de to berørte testfilene har 13 PASS. En separat lesende
  full Fold2-kontroll fant eksakt M1-rad ved 47 327/47 338 fit-issues og
  17 288/17 288 senere score-issues. M5-side var entydig ved bare
  16 166/47 338 fit-issues og 5 743/17 288 score-issues; bit5 var
  27 662 og 10 373. Samme kildeoppslag gir entydig fit-side på
  M15/H1/H4/D1 ved henholdsvis 18 268/18 158/19 110/16 361 issues.
  Dette viser bred historisk klokke-/kildedekning, men ikke ONLINE-
  levering, modellkonsum eller Edge. Bit5 er en målt målgråsone, ikke
  manglende M1-bar.
- En lesende metadata-binding av den kvalifiserte Fold2 TA-spanvisningen
  mot gjeldende kode og `direct_tf_hidden=True` besto med fem egne S/F-
  sidehoder. Den beholder 317 lokale felt og 255 felt per TF;
  kontraktshash er `29487410165986cfba333749a7ebbd1875eea0e93d710dbfbb6f57093fac94f5`.
  Dette er ikke lagret modellinput, fysisk forward eller gradientbevis.
- Strukturkandidatens framtidige pivotbekreftelse kunne tidligere telles selv
  når bekreftelsesbaren først lukket nøyaktig på foldens eksklusive slutt.
  En målrettet test feilet før rettelsen og består nå: modning krever
  `bekreftelsesbar_start + 5 min < foldslutt`. Alle ni fokuserte
  strukturtester består. Den gamle fysiske Fold2-kjøringen brukte den
  tidligere kildeversjonen; dens 47 195 kjente parklasser er historisk
  kildedekning, ikke kvalifisert dekning for den korrigerte målsemantikken.
  Ikke relanser den konsumerte planen; beregn korrigert, bundet dekning
  som del av én samlet kampanje.

- Serving-kilden har allerede en gjenbrukbar M1-bane: `PrebuiltServingSnapshot`
  holder hashbundet `BASE28` med rå M1-OHLC, og
  `v12_state_from_prebuilt.get_closed_m1_bar` krever eksakt minutt,
  samme pargenerasjon og uendret kildefilstempel. Leseren merker raden
  `complete=True`; faktisk lukking ved beslutning krever paritetskontroll. En M5-
  beslutning ved `t` kan dermed be om de fem lukkede M1-barene med starter
  `t−5` til `t−1` minutter fra én serving-kapsel. S-tapens gyldighet
  `t` svarer til slutt på M1-baren `t−1`. Dette er kodelesning av en
  tilgjengelig leveringsvei, ikke bevis for faktisk live lukking;
  gjeldende ONLINE-modell konsumerer ennå ikke banen.

- En capped 4G lesende Fold2-kontroll sammenlignet hashbundet S-TRAIN-M1
  med den arkiverte, hashbundne Entry-serving-parets `BASE28` på alle 47 338
  fit-issues. Fem lukkede M1-minutter per M5-issue ga 235 956 observerte av
  236 690 plasser. Alle 13 mid-/bid-/ask-OHLC- og volumfelt var bitidentiske
  ved observerte plasser; kilde-/tilstedeværelsesmasker og S-tapens siste
  M1-kildeindeks hadde null avvik. 47 124 issues hadde full femminuttersbane,
  214 var ufullstendige og skal maskeres uten å droppes. Aggregatet og
  kildehashene står i `docs/TA_M1_PATH_ARCHIVE_PARITY_REVIEW_20261116.json`.
  Dette er arkivparitet, ikke fysisk live-levering, TA-modellinput eller Edge.
- En ny ren `M1IssuePathTape` returnerer ved hver M5-issue akkurat de fem
  M1-startene `t−5..t−1`, fire mid-OHLC-felt i bps fra siste M1-slutt,
  observasjonsmaske per minutt og full-baneflagg uten radfjerning.
  Tre fokustester består; en capped 4G fysisk kontroll på alle 47 338
  Fold2-fit-issues ga samme 47 124 fulle baner og 235 956 observerte
  plasser fra både S-TRAIN-M1 og arkivert `BASE28`, med bitidentiske
  normaliserte input og masker. M5-intern rekkefølge blir nå representerbar;
  lengre M1-historikk for øvrige TF og faktisk ONLINE-levering er ennå ikke
  bevist; den rene kildekontrollen målte ikke modellforward eller Edge.

- En ny avskrudd-som-standard `IntegratedTAMarketStateModel` binder samme
  fulle TA-encoder, fem separate post-attention TF-vektorer og egne S/F-
  retningshoder per TF til den maskerte fem-M1-banen. En nullinitialisert,
  TF-lokal port bevarer eksisterende S/F-output eksakt ved start; M5-struktur
  har egne logitter for treff av frossen topp og bunn, og ukjent fasit maskeres.
  Tre syntetiske fokustester består for ett encoder-kall, startparitet,
  TF-lokal følsomhet, gradientvei og ukjent-maske. Dette er en kildekandidat:
  denne kildekontrollen alene viste ingen faktisk TRAIN-forward eller læring;
  den fysiske kontrollen nedenfor lukker bare forward-/gradientgapet.


- Én capped 4G, lesende Fold2-preflight på 16 faktiske TRAIN-rader besto med
  to modellforwards og null optimizersteg/checkpointlesninger. Nullporten ga
  bitidentisk S/F-output mot eksisterende dualmodell. Alle fem S- og fem
  F-sidehoder hadde ikke-null gradient på kjente fasiter; M1-port og
  topp/bunn-hode hadde også ikke-null gradient. Valgt batch hadde 16 fulle
  fem-M1-baner og begge strukturfasiter observerte på 16/16 rader.
  Korrigert strukturkilde ble samtidig målt på Fold2 TRAIN: 47 608 kjente
  topputfall, 47 619 bunnutfall og 47 195 kjente parklasser. Den første
  preflighten stoppet på manglende readonly-merke, ble rettet; en bestått
  mellomliggende batch manglet D1-fasit og ble fulgt av en batch med kjent
  S/F-side på alle fem TF. Eksakte kvitteringer og begrensninger:
  `docs/TA_INTEGRATED_PHYSICAL_PREFLIGHT_REVIEW_20261116.json`.
  Dette beviser numerisk kobling, ikke læring, ONLINE-levering eller Edge.


- Et nytt privat, hashbundet Fold2 TRAIN-opptak binder M1-bane og korrigert
  M5-topp/bunn-fasit til alle 47 338 fit-issues. Det gjenbruker de samme
  235 956 observerte M1-plassene og 47 124 komplette banene som kildepariteten.
  Toppfasit er observert på 47 193 rader, bunnfasit på 47 188, begge på
  47 152. Arkivet er 3,43 MB, ligger i GX1_RUNS og er ikke pushet.
  Kjøringen var capped 4G, null modellforwards og null optimizersteg;
  kvitteringer og hash står i
  `docs/TA_INTEGRATED_FOLD2_FIT_INPUTS_REVIEW_20261116.json`.
  Den fysiske modellpreflighten bandt viewhash `8056fe96...`; den tidligere
  metadatahashen `29487410...` er ikke identisk med denne og må ikke brukes
  som kvittering for den fysiske modellen. Senere score-input og samlet
  tapsoppskrift er ennå ikke bundet.


- Én kildekontrakt for samlet Fold2 S/F/struktur-tap er lagt til. Den bruker
  eksisterende fem-TF aktivitet og kjent side med M5 q/N-observasjonsvekting,
  F-vekt 1,0 og et forhåndsvalgt mindre hjelpeledd på 0,25 for maskert
  topp/bunn-BCE. To fokustester består for aritmetikk og null gradient på
  ukjent strukturfasit; faktisk optimizersteg er ikke kjørt. Det faste
  4 096-raders senere utviklingspanelet (januar–mars 2025) er nå bundet til
  et privat M1-/strukturarkiv: 4 089 fulle fem-M1-baner, topp/bunn observert
  på 4 068/4 081 rader. Dette er gjenbrukt utvikling, ikke forseglet TEST
  eller uavhengig bekreftelse. Kvitteringer og begrensninger står i
  `docs/TA_INTEGRATED_OBJECTIVE_LATER_INPUTS_REVIEW_20261116.json`.


- Den samlede tapsfunksjonen er kjørt én gang på faktisk Fold2 TRAIN-input
  med samplet S-spørring, native F, korrekt M1-/struktur-tape og alle fem
  tidsrammer. Alle ti sidehoder, M1-porten og strukturhodet hadde ikke-null
  gradient. Total tap 2,825355 = S/F 2,590821 + 0,25 × struktur 0,938137.
  Null optimizersteg, to modellforwards; CPU-initialdigest er bare mekanikk,
  ingen fersk CUDA baseline. Kvittering:
  `docs/TA_INTEGRATED_ACTUAL_OBJECTIVE_PREFLIGHT_REVIEW_20261116.json`.
  Fit, senere score, kalibrering, vending, kostnader og Edge gjenstår.


- Den eksisterende Fold2 fit-/score-inputruten kan nå bruke separat TF-dekoder
  uten å ommerke den gamle senere sidecar-kvitteringen: en streng ombinding
  krever at alle fysiske viewfelt er like og at bare de to dekoderflaggene
  endres. På én faktisk senere utviklingsbatch ga en capped 4G CPU-kontroll
  ni finite TA-flater med 317 lokale felt og 255 per TF. Testene for gammel
  inputeier og ombindingen besto. Dette er kilde-/gatherparitet, ikke
  modellscore eller prediktiv gevinst. Aggregat:
  `docs/TA_INTEGRATED_LATER_VIEW_REBIND_REVIEW_20261116.json`.

- Én integrert ikke-GC Fold2-plan er nå fysisk bundet til samme fullstendige
  TA-input, egne S/F-hoder på fem TF, fem lukkede M1-barer og maskert
  M5-topp/bunn-tap. Plan-SHA er
  `4016255217a3c87bdc84715b6e65707090fa09dcaa1e70a3a1b3455bdac84621`.
  Kilde-/input-preflight besto uten modellkjøring eller CLAIM. En capped
  4G-kontroll av én faktisk senere B16-batch bekreftet initialmodellens
  digest og at skåringen bruker eksakt Fold2 S/F-tapskontrakt; én forward,
  null optimizersteg. Maksimum ved eventuell separat innrømmelse er
  47 338 TRAIN-rader, 2 959 oppdateringer, 1 024 initial-/final-
  skårforwards og 10 500 sekunder under 20G/512M/GPU-vakt.
  Fast senere M5-vendingslesning er nå definert på eksakte nabobarer:
  kjent forrige S-side som nåfasit, nåværende issue sin S-prognose for
  neste fem minutter, AUC/Brier og parvis dag-/ukekontroll uten terskelsøk.
  Leseren gjenskapte et annet, allerede lagret Fold2-opptak med 531 kjente
  M5-par og AUC 0,457503 → 0,541573; H4 hadde null gyldige nabopar.
  Dette validerer leseren, ikke den nye modellens læring eller Edge.
  Se `docs/TA_INTEGRATED_FOLD2_FIT_PREPARATION_20261116.json`.
  Den eksakte engangsadgangen sto i
  `docs/TA_INTEGRATED_FOLD2_FIT_ADMISSION_20261116.json`. Etter fysisk boot
  656 feilet kjøringen i min nye `structure_readout`: boolske hit-etiketter
  ble kastet til float64 før en AUC-funksjon som indekserer med bool.
  256 initiale TRAIN-forwards, null optimizersteg, ingen checkpoint eller
  senere score. Terminal exit 1 og vaktlogg er bundet i
  `docs/TA_INTEGRATED_FOLD2_FIT_FAILED_REVIEW_20261116.json`.
  Ingen ny plan er innrømmet.
- Den konkrete feilen er rettet med én boolsk etikettlinje og en fokusert
  regresjonstest. En capped, lesende kontroll besto hele skårveien på
  faktiske TRAIN-/senere B16-blokker med begge topp/bunn-klasser, samme
  initialmodell og null optimizersteg. Ny plan002 har samme inputs,
  modell, tap, rekkefølge, panel og grenser som konsumert plan001;
  bare rettet kilde og ny engangsidentitet skiller. Plan-SHA er
  `330e4cb1dd4c76980264b44a32f091e67c43142819514901f60068abed102783`.
  Kilde-/input-preflight besto med null forwards/CLAIM, og runner avviser
  eksplisitt plan001 med exit 75. Se
  `docs/TA_INTEGRATED_FOLD2_FIT_002_PREPARATION_20261116.json`.
  Operatøren presiserte 16.11 at nytt «ja» per stor lokal kjøring skal bort.
  Integrator innrømmet deretter plan002 under stående 0 USD-fullmakt; fysisk
  boot, ressursvakter og øvrige grenser ble bevart. Den fullførte med
  terminal exit 0 og er nå konsumert. Eksakte kvitteringer:
  `docs/TA_INTEGRATED_FOLD2_FIT_002_TERMINAL_REVIEW_20261116.json`.

- Lesende full Fold2-målkontroll: eksisterende fem-endepunkt-S og M1-
  spørringstape er identiske på alle 236 690 fit-celler (null tillatelses-
  eller modenhetsavvik). Faste native S-spørringer gir kjent side på M5
  16 101, M15 18 182, H1 17 973, H4 18 391 og D1 12 291 rader. I
  fit002 var native kjent S-side spurt på bare 16 101/6 112/1 488/340/37;
  høyere TF hadde likevel mange kjente sider ved andre spørringshorisonter.
  Endringen gjelder native endepunktsupervisjon, ikke nye fasiter. Direkte
  motsatt S/F-side finnes i 2,4–9,0 % av felles kjente rader per TF.
  M5/D1-fasit er derimot ulik i 2 072/4 234 S- og 2 689/6 057 F-rader;
  nesten felles fem-TF-output kan ikke representere dette. Ingen ny
  merkemekanikk, modellforward eller fit ble brukt. Eksakt målscope:
  `docs/TA_NATIVE_S_F_TARGET_SCOPE_20261117.json`.
- En forhåndseksisterende Fold3 Apr–jun 2025 utviklingsakse har 17 537
  rader og kjent native S-side på 5 734–6 663 rader per TF; F har
  5 528–6 886. Bare mål er kvalifisert. TA-featureinput, lukket M1-bane
  og strukturfasit må bindes før en ny kronologisk score. Dette er
  utvikling, ikke forseglet bekreftelse eller Edge.

- Apr–jun 2025-utviklingsinput er nå eksakt forberedt: 17 536 komplette
  B16-rader av 17 537 på forhåndsvalgt M5-akse. Eksisterende seksflaters
  TA-produsent får målt fysisk behov på lokal/M5/M15/H1/H4/D1
  17 697/17 617/5 931/1 563/480/316 rader. Ny visning endrer bare
  kildehash og avledet visningshash; Fold2-fasiter, features, fullakser og
  normalisering er uendret. Kvalifisert plan-SHA
  `906ec2f69a6208bb9f4b83f6dbb7cb1ce6eaf5eb6fa35e8d65e00a9daa4b59d1`
  ble innrømmet én gang som 4G/512M, maks 2 700 sekunder, bare lesende
  inputproduksjon. Den fullførte med terminal exit 0 på 319,13 sekunder:
  seks flater, 31 gamle felt per flate med null float32-avvik, 17 536
  lokale og 87 680 MTF-vinduer. Kilde uendret, null modellforward/fit,
  TEST ubrukt. Engangsadgangen er konsumert. Eksakte RESULT/COVERAGE/
  TERMINAL-ref står i policy. Dette er fysisk TA-input, ikke en faktisk
  normalisert modellbatch eller Edge.

- M1-/strukturplan001 SHA
  `15d534feed3cf3852717a7add6e736804789544517b145028944cee14c088748`
  ble CLAIM-et én gang og feilet etter 4,96 sekunder før output: closure-
  filens SHA stemte, men planens binding hadde tre metadatafelt i tillegg
  til path/SHA; ny leser krevde eksakt tofeltsref. Ingen modellforward,
  optimizersteg eller M1-/strukturarkiv. Terminal og FAILED er bevart i
  privat RUNS og referert fra policy; plan001 er konsumert.
- Minste rettelse normaliserer closure-ref til path/SHA og kontrollerer den
  allerede i preflight. Ny plan002 SHA
  `a200d63f4b128d89b7a282bf20a7b6b39a3d57c09237caad0f4b7bfd8dc0c445`
  hadde samme 17 536 Apr–jun-rader, 31 M1-/21 M5-kildegrupper og 4G/512M,
  maks 1 800 sekunder. Den fullførte med terminal exit 0 og uendret kilde.
  Arkivet har `(17 536,5,4)` lukket M1-bane med 17 470 komplette rader;
  ukjente M1-plasser er nullmaskert. M5-topp/bunn-fasiten er observert
  på begge sider i 17 399 rader, og treff uten observert fasit er null.
  Arkiv-SHA `1d7163801d5fb1eb5a43e48511d79ef7d2199d16c3655d3e5cba8fb4d77086da`.
  RESULT/TERMINAL står i policy. Engangsplan002 er konsumert; null
  modellforward, optimizersteg eller TEST. ONLINE-paritet og Edge er ubevist.

## Ikke gjort / faktisk hindring

Den avsluttede fit002-lesningen er bundet i
`docs/TA_INTEGRATED_FOLD2_FIT_002_RESULT_REVIEW_20261116.json` (SHA256
`685290f9080682c36d63bba69ba45f03954fd446743b127a1c062cb04d3e298e`).
Senere S/F-side-Brier var dårligere enn fryst TRAIN-prior
i 10/10 TF-/oppgaveceller; treklasse-Brier var også dårligere i 10/10.
Struktur topp/bunn slo prior med AUC 0,7945/0,7834, men M5-vending på
531 kjente nabopar fikk AUC 0,4535. F-hodene er fysisk separate, men
deres retningsoutput er fortsatt sterkt korrelert. Strukturfasiten gjelder
neste bekreftede pivot (median 9 M5-barer til bekreftelse), ikke neste
M5-vending. Kilden beviser dette målskillet; den eksakte årsaken til
svak sidelæring er fortsatt uisolert. Senere panel er gjenbrukt utvikling.

En lesende TRAIN-kontroll av samme låste modell og mål viste M5-vendings-AUC
0,4656 på 551 kjente nabopar (INITIAL 0,4625). Hele fit-populasjonen har
6 746 eksakte kjente M5-sidepar, hvor 3 547 skifter; alle de aktuelle
M5-beslutningene hadde komplett lukket M1-bane. S fikk native kjent
sidefasit på M5 16 101, M15 6 112, H1 1 488, H4 340 og D1 bare 37
rader. F hadde minst 15 148 kjente native sider per TF, men outputene
forble sterkt korrelerte. Dette er to ulike hindringer: sparsom native
S-eksponering på høyere TF og F-kollaps tross dekning. En ekstra
vendings-BCE på samme M5-sidelogit er algebraisk bare omvekting av
rader med kjent nåside, ikke ny informasjon eller en begrunnelse for
et nytt hode. Ingen ny trening eller VAL-utvelgelse fulgte.

En capped, lesende CPU-forward på 16 faste TRAIN-rader gjenskapte lagret F-logit
innen 0,000014. De fem TF-representasjonene og F-projeksjonene var ulike
(M5/D1 relativ avstand henholdsvis 0,448 og 0,338), men F ga likevel
likt sidefortegn på 14/16 av disse radene. På hele det faste TRAIN-panelet
hadde M5/D1 ulik kjent F-side i 211/517 tilfeller; F-outputene var ulike
i bare 36. Senere utvikling ga 249/557 mot 15. Den effektive kollapsen
ligger dermed etter ulike TF-vektorer, i F-utlesning/læring. Årsaken
(tapskonflikt, kronologisk rekkefølge, utilstrekkelig fit eller svakt
signal) er ikke isolert. Ingen ny optimizer eller model fit ble kjørt.

En lesende dekomponering av de hashbundne faste TRAIN-prognosene viser at F
undervurderer OPP på alle fem TF: kjent fasit er 51,1–64,7 % OPP, mens
modellens gjennomsnittlige P(OPP) er 33,6–43,2 %. På M5 er F-Brier
0,255209 mot fryst TRAIN-prior 0,249876; 0,006283 av feilen kommer fra
kvadrert avvik mellom gjennomsnittlig prognose og fasit, mens prediksjon/utfall-samvariasjon motvirker noe.
En rent algebraisk fjerning av dette middelavviket på samme TRAIN-panel ville gi
M5-Brier 0,248926, under prior med mindre enn 0,001; den endrer ikke
rangering eller M5-vending. M15/H1/H4/D1 er fortsatt svake ved slik optimistisk
TRAIN-diagnose. Full TRAIN-fasit under faktisk batchnormalisering er over
50 % OPP på alle fem TF, og kilde-/loss-/skårtegn for OPP er samme bit4.
Batchvekting eller enkelt fortegnsbytte forklarer derfor ikke F-kollapsen.
Dette isolerer ikke optimizer, oppgavemiks, representasjon eller signal.
Ingen ny fit ble kjørt.

En første capped lesende F-representasjonskjøring nådde 4 096 TRAIN-rader, men
feilet på en for trang CPU/GPU-paritetsgrense på 0,000020; planen er konsumert.
Ny engangsplan fullførte 256 CPU-forwards på samme panel. Største lagret
F-logittavvik var 0,00002313, middelavvik 0,00000441, og modelldigesten
var uendret. RESULT.plan.path ble feilaktig siste M1-tensor fordi privat
auditkode gjenbrukte variabelnavnet `path`. Originalkvitteringen er bevart;
CLAIM, terminal og en separat RECONCILIATION.json validerer opptaket
SHA256 `d4c8fa49b82dd6095afa17ebfdf40aff2bcf00b9245450edc49044d8325fb256`.
Detaljer og begrensninger står i
`docs/TA_FOLD2_F_FROZEN_READOUT_ADMISSION_20261117.json`.

Den eksakte CPU-kontrollen med PLAN SHA256
`c87fcec1c25e351fec6e9c7a7bf2905ce97605661ba9055cc769e621b757e786`
fullførte med exit 0 og ti faste lineære F-utlesningsfit på frosne
representasjoner, uten native modellforward eller optimizersteg. Den er
konsumert og training_enabled er lukket igjen. Første 3 072 kronologiske
TRAIN-panelrader ble brukt til fit, siste 1 024 til score, med modne labels
og embargo. På fit-radene fikk projeksjonsutlesningen AUC 0,668–0,974,
men på senere TRAIN-rader var alle ti Brier-skårene dårligere enn både
fit-prefix-prior og opprinnelig F-hode. M5 projeksjon fikk score-AUC
0,510 og Brier 0,27152 mot prior 0,24971; D1 hadde sterk klassedrift
(70,7 % OPP i fit, 45,5 % i score). Lagrende parametre og prognoser er
rekonstruert uavhengig med samsvar i AUC/Brier innen 1e-12.
Se `docs/TA_FOLD2_F_FROZEN_READOUT_RESULT_REVIEW_20261117.json` (SHA256
`01e4ed18528116681b8d808f6c5ba28906c395432d5e5d29ae697c0f4d457cf2`).
En enkel lineær nyutlesning er ikke tilstrekkelig reparasjon her. Encoder
og normalisering hadde allerede sett hele Fold2 TRAIN, også scorekvartalet;
resultatet er en mekanikkontroll, ikke uavhengig OOF eller Edge. Ingen
ridge-tuning på scorekvartalet følger.

Det målte problemet er svake M5-vendinger og utilstrekkelige selvstendige
fem-TF-retninger. M5-only-fitten gir begrenset senere S-rangering, men
ikke sikker før/etter-gevinst eller bedre Brier enn prior; F og faktiske
M5-vendinger forblir utilstrekkelige. Lineær lesning før pooling ga heller
ingen M5-reparasjon. At M5-side var ett av 25 sidemål i den
separate full-GC-fitten kan derfor ikke alene forklare dette. Den historiske
nå-/framtidssidefasiten er ofte uavklart, og den trente TA-modellen gir
nesten felles ikke-M5-logitter tross tilkoblet TF-koordinat. Dette er målte
begrensninger i mål og output; deres andeler av M5-feilen er ikke isolert.
S-fasiten er fem marginale framtidstilstander per TF, ikke direkte vendetid
eller kostet Entry.
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

Parede S-spørringer er nå kodet for én samlet ikke-GC-kampanje: ett
encoderpass gir både label-blindt samplet S og native S på alle fem TF, med
separat native F og M5-struktur. Fast tap er 0,5 samplet S + 0,5 native S
+ 1,0 F + 0,25 struktur. Dermed får de tidligere nesten fraværende
native H4/D1-sidene gradient uten at M5-S dubleres. Sju fokuserte
modell-/tapstester består, inkludert nullgradient for ukjente mål og
M5-paritet. Ny Fold2-normalisert april–juni-visning og én felles inputadapter
binder 47 338 TRAIN-rader og 17 536 faste score-rader; adapterens faktiske scorebatch er kjørt. Engangs-CPU-preflight plan SHA
`54c4e9b8670d2103afadd32798248695678bd6a6b24823a8e3b503e5490c9f8f`
fullførte med terminal exit 0 på 12,4 sekunder. Åtte numeriske flater ble
normalisert med Fold2-frosne verdier; kategori var den niende flaten.
Paret S/F/struktur-forward og masket tap var endelig, med kjente mål på alle
fem TF i denne ene batchen. Modellens CPU-initialdigest var identisk med
tidligere integrert initialtilstand; null optimizer ble konstruert eller kjørt.
Resultat SHA `24ae40f68edcf15778081ac703c8f423e9b155f8d5fc4e88c14c56736e14418f`.
Gaten er konsumert. Ingen Edge eller kampanjefit er påstått.

En ny, ennå ikke innrømmet engangsplan er bundet til ett Fold2 TRAIN-pass
(47 338 rader, høyst 2 959 optimizersteg) og initial/slutt-skår på alle
17 536 faste Apr–jun 2025-utviklingsrader (2 192 score-forwards). Plan SHA
`855bc0abb4e8a8a375bb126ec82f3afa3d40b5ac17b335f67d652793b370f5b3`.
Måltap, fem TF-hoder, fryst Fold2-prior, F-kalibrering, kjente M5-nabovendinger,
TF-uenighet og struktur er forhåndsbundet. Kjøringen har 10 500 s
skriptfrist, 10 800 s ytre vakt, 20G/512M og signerte GPU-grenser.
Fersk fysisk Windows boot-ID 658 er kontrollert. Metadata-preflight på ren
kilde besto uten CLAIM eller modellforward. Bare denne planen er nå innrømmet
for én voktet lokal kjøring; trening er åpen kun til den konsumeres.
Ingen skår eller økonomisk Edge er hevdet fra planen.

Engangskampanje plan001 ble CLAIM-et etter fysisk boot 658, men stoppet
etter 11 sekunder før modellforward, optimizer, checkpoint og score. Min nye
inputreferanse hadde nøkkelen `coverage`, mens planen ventet
`physical_coverage`; terminal exit 1 og `KeyError: coverage` er bevart i
`/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_PAIRED_FIT_20261117_001/TERMINAL.json`.
Plan001 er konsumert. Signert GPU-vakt målte ingen belastning utover idle.
Trening er stengt igjen. Dette er en mekanisk koblingsfeil, ikke et målresultat.

Den eneste rettelsen etter plan001 er én felles inputref-funksjon med
`physical_coverage` og `issue_result`, kontrollert også i metadata-preflight.
Distinkt plan002 SHA
`10daa0b49a18f72abebfeb3549ca4330986209adf473685e2afe5c4ce8850839`
har samme modell, mål, data, tap, score og ressursgrenser, men ny kilde og
engangsidentitet. Metadata-preflight besto med eksakt inputref-nøkkelsett, null CLAIM og
null forward. Fysisk boot-ID 659 er kontrollert. Kun plan002 er nå innrømmet
for én voktet kjøring; plan001 forblir urørt og konsumert.

Plan002 fullførte etter fysisk boot 659 med terminal exit 0 og uendret kilde.
47 338 TRAIN-rader ble besøkt i 2 959 batcher og alle 2 959 optimizersteg
ble utført. Den faste Apr–jun-perioden fikk 1 096 initiale og 1 096
endelige skårforwards, 17 536 rader hver. Resultat SHA
`9ae36908fcbbc95f4170f0ed90231aa7be61bfcd163efd780b1e0bcf18fe0b1a`,
terminal SHA `c1bcc24963d5f3a86cce23e6448e38b3bd2158af7e15588b97d06f2a6072256b`.
Skriptet brukte 7 348 sekunder; signert vakt topp 54 °C kjerne,
58 °C minne, 165,97 W og 2 172 MiB. Checkpoint og råopptak ligger kun
under GX1_RUNS; ingen vekter er pushet. Engangsplanen er konsumert og
training_enabled=false. Den faste skåringen er kontrollert mot de SHA-bundne initial- og
sluttopptakene. Brier, observerte antall og M5-vendinger ble regnet om fra
opptakene uten ny modellforward. Sluttresultatet er en negativ kontroll for
native M5-retning: S-side Brier 0,253992 mot fryst TRAIN-prior 0,249572
(AUC 0,516; 5 734 kjente), F-side 0,261731 mot 0,249882 (AUC 0,498;
6 886 kjente). F-side er svakere enn samme prior på alle fem TF; bare
D1 native S-side er marginalt bedre (0,242887 mot 0,245305). Av 2 265
kjente sammenhengende M5-par hadde 1 138 faktisk sideskifte. Modellen
meldte 580 skifter, hvorav 273 riktige og 307 falske: presisjon 47,1 %
mot faktisk skifteandel 50,2 %. Fem TF-output er mer ulike etter fit,
men native retningskvalitet er ikke kvalifisert.

Strukturhodet lærte den avgrensede fasiten «treffer neste bekreftede
M5-topp/bunn et nivå fryst ved beslutningstid?»: høy AUC 0,843,
Brier 0,162625 mot prior 0,250109; lav AUC 0,836, Brier 0,171595
mot prior 0,248721. Dette er framtidige pivotutfall, ikke neste
prisretning eller kostet Entry. En eksplorativ, naturlig forskjell mellom
sannsynlighetene for høy- og lavtreff ga AUC 0,488 mot kjent native M5-S
(5 687 rader) og 0,510 mot M5-F (6 832 rader). Den binder derfor ikke
strukturtreff til en brukbar M5-sideprognose.

Samplet S-spørring ga høyere AUC på M15/H1/H4/D1 (0,679/0,720/0,739/
0,682), men spørringsavstanden varierer fra 5 minutter til tidsrammens
native endepunkt. På delmengden der kausal nåretning og framtidig side
begge er kjent, ga enkel videreføring av nåretningen AUC 0,638/0,697/
0,755/0,757 på M15/H1/H4/D1; modellen ga 0,689/0,734/0,755/0,709
på nøyaktig samme rader. H4 er dermed lik referansen, D1 svakere;
M15/H1 har mulig kortsiktig merverdi som ikke kvalifiserer M5-vendinger.
Disse delmengdene og den eksplorative strukturkontrasten er diagnostikk,
ikke et forhåndsbundet handelsutvalg. Ingen uavhengig handelsøkonomi,
generalisering eller Edge er bevist.

En separat, ennå ikke innrømmet kost-/prisobservasjon er nå kildebundet til
plan `TA_APR_JUN_M5_FIXED_COST_20261117_001` (SHA
`d92d6108146328aaec3bfaace0a6f5f4d6df6d2fb1a77472f6d41346ef1cf9ed`). Den bruker bare sluttprognosene for M5-F fra
plan002, samme 17 536 Apr–jun-rader, eksisterende M1-quote-kilde og
oppgitte 4 bps tur-retur-kostnader. Før prisutfall leses, er seks
observasjonspunkter (1/5/15/60/240/480 min), treklasse-argmax med FLAT
ved likhet, tvunget OPP/NED som diagnostikk, og alltid-LONG/SHORT/FLAT
bundet. Samme komplette prisbane-maske brukes for alle. Dette er ingen
holderegel, Exit, handelsportefølje eller ny modellkjøring. Eksakt
4G-cappet preflight besto på ren kilde uten CLAIM, prisutfall eller
modellforward. Den ene CPU-auditen er nå fullført og konsumert; trening
og TEST forblir stengt.

Engangsauditen fullførte på 54,9 sekunder med null modellforward og null
optimizersteg. Resultat SHA `1bed7bdc818e89c5b9163c802d941c680f0f939de2a7b2cc403d6fe07521cf8d`, terminal SHA
`56651c04d1e2a36148a93d44d0457c5f4c1b8fad63b05c7f1bbcf1c38feda870`. Den forhåndsbundne treklassebeslutningen fra
native M5-F valgte FLAT på samtlige 17 536 prognoser; på felles pris-/
banemaske var den identisk med alltid-FLAT ved alle seks målepunkter.
F-fasiten i opptaket består av 10 650 FLAT, 3 368 NED og 3 518 OPP:
60,7 % FLAT. F defineres som fortegnet til lukket M5-prisforskyvning
ved native endepunkt bare når størrelsen når 0,5 × ATR. Den er verken
kostet Entry-verdi eller konservativ MAE før MFE. Modellens gjennomsnittlige
aktivitetsprognose var 0,380; tvungen side ga 1 632 LONG og 15 904
SHORT på hele opptaket.

Tvungen side hadde negativt gjennomsnitt etter oppgitte kostnader ved
alle seks faste observasjonspunkter: −6,146/−6,196/−6,168/−6,396/
−6,692/−3,766 bps ved 1/5/15/60/240/480 min. Ved 480 min slo
−3,766 bps både alltid-LONG (−7,872) og alltid-SHORT (−4,933) på
samme 11 368 komplette rader, men tapte fortsatt i absolutt forstand.
Alle punkter rapporteres; ingen horisont ble valgt som holderegel.
Kostoppsettet er 4 bps tur-retur pluss deklarert finansiering, ikke
historiske fyll. Overlappende markouts er ikke posisjonsregnskap,
selektiv handelsstrategi, uavhengig generalisering eller økonomisk Edge.

Kildediagnosen er nå avgrenset uten ny fit: F-M5-merket har eksakt
femminuttersendepunkt, ingen mellomliggende kildegap på modellradene og
bitlik M5-pris/ATR14-merkeproduksjon
(`docs/TA_F_M5_TARGET_ENDPOINT_GAP_REVIEW_20261114.json` og
`docs/TA_F_M5_PRICE_ATR_SOURCE_RECONSTRUCTION_20261114.json`). Fem
lagrede 64-runders F-tabularmodeller lærte TRAIN-radene (AUC 0,699–0,795),
men fikk bare 0,501–0,513 på senere foldrader og tapte Brier mot frossen
TRAIN-prior. Indre TRAIN-kontroll gjentok overføringssvikten
(`docs/TA_F_M5_SAVED_MODEL_TRAIN_SCORE_DIAGNOSIS_20261114.json` og
`docs/TA_F_M5_INNER_TRAIN_TRANSFER_RESULT_REVIEW_20261114.json`). Den
nye integrerte fitten fikk F-M5 AUC 0,498 og dårligere Brier enn prior.
En eldre, separat oktoberkontroll av F-retning mot hvilken LONG/SHORT-side
som fikk bedre netto +5-minutterspris ga AUC 0,491 på 6 297 rader;
dette er et annet kohort og beviser ikke Apr–jun-økonomi. De undersøkte endepunkt-, pris-, ATR- og gapfeilene er dermed
avvist, mens årsaken til svak overføring ikke er isolert. F er korrekt som separat terminalt prognosemål, men kvalifiserer
ikke en Entry-handling. Flere F-only-epoker, hodesplitter eller
indikatorregler uten nytt beslutningsmål har ingen målt begrunnelse.

M5-kost-/prisbanemålene fra engangsplan
`TA_M5_COST_PATH_TARGETS_20261117_001` er nå fullført og konsumert.
Plan SHA `ac265434e616bc7cbfc6e1cca122d083f0e40cc13b1cd4f6520e1671c67b523e`;
CLAIM SHA `d6ef09efd9e60ef0632fb65e2cf1a5a4b4efffcc3bcdf8af193c53a7538c12b1`,
RESULT SHA `dd8741c5f4d757a4ce366f681ff808841f20ca59f5e663413280b26aa2953ee7`,
TERMINAL SHA `d6334b04ce9ada129291cb9244d2ec131aaafb4084a9aad550e59eeceaf114a6`.
Kjøretid etter CLAIM var 56,6 sekunder. FIT-arkivet har 47 338 rader
(SHA `3f0a9c74a4a40da80fc62850b62dde3b67d7f896963440d58a5c810af9908024`);
SCORE har 17 536 rader (SHA `3969494002f84a913161072da9f77abea86b82e1213d8b257468a7eb71f18a34`).
Begge har separate terminal- og komplett prisbane-masker samt LONG/SHORT
netto, MFE og konservativ MAE før MFE ved alle seks forhåndsbundne
observasjonspunkter. Uavhengig kontroll fant eksakte inputakser,
filhash, maskeinvarianter og FIT-paritet mot tidligere M1-kostaudit.
SCORE-kostgjennomsnittet er bitlikt for LONG og SHORT ved alle seks
punkter når samme komplette prisbane-rader brukes. Ved 480 minutter
har 15 480 rader sluttpris, men bare 11 368 komplett prisbane; disse
må ikke blandes. Bakoverskuende andel med minst én positiv side er
ikke modellens fangstevne eller Edge. Null modellforward, null
optimizersteg, ingen TEST eller handel. Treningsporten er stengt.

## Neste steg

1. Behold plan002, kostplan001 og målplan001 konsumert og treningsporten lukket.
   Ikke start flere F-only-epoker, hodesplitt- eller indikatorregler på
   teknisk PASS; den rene F-kilden og gjentatt transferfeil er dokumentert.
2. Bind én integrert ikke-GC M5-beslutningskandidat til de lagrede
   Fold2/Apr–jun-kost-/prisbanemålene: separat kostet LONG/SHORT/FLAT-
   verdi og konservativ prisbane-kvalitet ved 1 minutt til mange timer.
   Behold de fem selvstendige S/F-retningene og M5-struktur som egne
   prognosemål fra samme kausale encoder. Avklar masker, TRAIN-skala,
   output/tap, fast handlingsregel og alle målepunkter før eventuell fit;
   ingen horisont velges etter utviklingsskår. Dette er designarbeid,
   ikke Entry-trening eller ny kjøreautoritet.
3. Økonomisk Edge krever senere alle valgte handler og åpne posisjoner
   på en uavhengig periode. Full GC, TEST, ONLINE, broker og handel er
   lukket. Gamle engangsplaner relanseres aldri.

Opprydding: historiske overleveringsavsnitt er fjernet her og bevart i Git.
Ingen Python-fil er slettet; gjeldende referanse-, import-, test- og
eierskapsrevisjon beviser ingen frakoblet eier. DATA/RUNS og checkpoints
tilhører retention-eieren.
