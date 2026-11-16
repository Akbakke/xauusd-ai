# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier gjeldende status;
prosess- og terminalkvitteringer overstyrer prosa. Tidligere overleveringer
finnes i Git.

## Gjeldende status

TA_M5_STRUCTURE_FOLD2_AUDIT_COMPLETE_TARGET_EDGE_UNQUALIFIED.
Den tidligere ikke-GC S-lesningen er fullført og M5-Edge fortsatt ubevist.
Fold2 S-only full-GC 25-cellers engangsplan er fullført og konsumert.
training_enabled=false, native_training_enabled=false og
full_epoch_training_allowed=false. Full GC er ifølge operatøren pauset
inntil videre (bekreftet 16.11). Windows-oppgaven
GX1RandomAccessCampaignV2 er deaktivert. Ingen relansering, native modellfit,
TEST, Entry/Exit, broker eller handel er innrømmet.

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
  prior. Ingen ny modellkode, flaggpromotering eller fit følger her.

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

## Ikke gjort / faktisk hindring

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

## Neste steg

1. Plan003 er konsumert og full-GC S-hypotesen forkastet. Operatøren har
   trukket den uinnrømmede 30-epokers memoriseringsdiagnosen; ingen trening
   åpnes av dette arbeidet.
2. Respekter Full GC-pausen. Bruk eksisterende, allerede målte ikke-GC-
   resultater til å avgrense M5-retning og de fire øvrige TF-retningene uten
   ny modellkjøring. M5 er det etablerte beslutningspunktet; mål om de
   separate TF-prognosene faktisk tilfører M5 verdi før en kildeendring.
   Neste TA-recipe må eksplisitt velge egne S/F-sidehoder på alle fem TF,
   med M5 som beslutningspunkt. En separat kildekandidat for neste
   bekreftede M5-topp/bunn mot nivåer fryst ved beslutningen står i
   docs/TA_M5_STRUCTURE_OUTCOME_CANDIDATE.md. Den binder nivåidentitet,
   pivotens tre lukkede bekreftelsesbarer, nivåtreff/brudd/avvisning og
   sensur ved M5-klokkebrudd/foldgrense, uten å omdøpe dette til S eller F.
   Eksakte dokumenterte markedsstengninger kan nå passeres bare ved full
   intervallinneslutning; øvrige M5-klokkebrudd forblir ukjente.
   Kildekandidaten er implementert i
   gx1/audit/measure_ta_m5_structure_candidate_v1.py og åtte fokuserte
   syntetiske tester består. En konkret sensurfeil mellom bekreftet fryst
   nivå og M5-beslutningen ble rettet før denne overleveringen; bar på
   den eksklusive foldgrensen avvises. Den ene fysisk bundne Fold2 TRAIN-
   lesningen er fullført og konsumert under 4G/512M/900s uten modellsteg.
   Originale CLAIM/RESULT/TERMINAL ligger privat i
   GX1_RUNS/TA_M5_STRUCTURE_FOLD2_AUDIT_20261116_001; samlet aggregering
   og begrensninger er i
   docs/TA_M5_STRUCTURE_FOLD2_AUDIT_RESULT_REVIEW_20261116.json.
   47 567/47 784 beslutningsbarer har begge neste pivoter bekreftet;
   47 195 har entydig parklasse. Av 589 ukjente parklasser mangler 217
   komplett par og 372 har samme bar som topp og bunn uten observert intern
   rekkefølge. OPP=18 043, NED=14 514, BLANDET=14 581, LIKT=57.
   Kjent lukning forklarer 166 passerte M5-klokkebrudd; åtte forblir
   ukjente i innlest segment. Dette er dekning, ikke prognosegevinst,
   ONLINE-paritet eller Edge; strukturklassen erstatter ikke S/F.
   Planen relanseres ikke. Per-plan spørregel for små lesende CPU-auditer
   er fjernet fra AGENTS.md; store kjøringer krever fortsatt spørsmål.
   Neste nødvendige kontroll er om framtidig struktur faktisk kan
   predikeres bedre enn fryst TRAIN-prior og kausal persistens før
   eventuelt targetvalg eller modellfit. En fast lesende sammenligning er
   forberedt i configs/research/TA_M5_STRUCTURE_PERSISTENCE_20261116_001.json:
   prior fra M5-issues mai–august hvis begge pivoter var bekreftet før
   01.09; evaluering september–desember. SMC 0→OPP, 3→NED,
   1/2→BLANDET, 4→fryst prior; ingen modellfit. Capped preflight og
   syntetisk suksess-/feilvei besto. Den lille lesende planen er innrømmet
   én gang under stående lokal autorisasjon; ingen CLAIM eller prislesing
   ennå. Ikke fyll bit5 med oppdiktet side.
   Ingen Full
   GC-kode, checkpointdiagnose eller ny full-GC-hypotese nå.
3. En senere særskilt innrømmet, kort matchet kontroll må vise både M5-side
   mot fryst referanse og faktisk vendingsdiskriminering før mer trening.
   Senere kronologisk generalisering og alle valgte/åpne posisjoners økonomi
   trengs før Edge. TEST, broker og handel er fortsatt stengt.
4. Den innrømmede S-lesningen er fullført, gjennomgått og konsumert.
   Ikke relanser eller lag flere lineære decoder-runder på samme panel.
   Før-pooling alene ga ingen M5-reparasjon mot fryst prior eller native.
   Neste læringssteg må være særskilt innrømmet, kildekausalt og målt på
   faktisk M5-beslutning og selvstendige fem-TF-retninger.

Opprydding: historiske overleveringsavsnitt er fjernet her og bevart i Git.
Ingen Python-fil er slettet; gjeldende referanse-, import-, test- og
eierskapsrevisjon beviser ingen frakoblet eier. DATA/RUNS og checkpoints
tilhører retention-eieren.
