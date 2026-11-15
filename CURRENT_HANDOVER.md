# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

Gjeldende policy: S_M5_JOINT_COST_ADMITTED_READONLY.
Ny separat engangsplan er innrømmet for en lesende, capped fellesmaskeaudit
av terminal netto, MFE og konservativ pre-MFE MAE på de samme S/M5-radene.
Faste Fold2 fit47338/senere4096 og seks forhåndsvalgte horisonter gjenbrukes.
Dette er etterpåkjent S-side som måldiagnose, ikke en modellbeslutning.
0 tillatte modellforwards, optimizersteg, TEST, broker og handel.
Ny plan: `configs/research/MARKET_STATE_S_M5_JOINT_COST_20261115_001.json`.
Forrige policy: S_M5_PATH_COST_DIRECT_S_SIDE_NEGATIVE_EDGE_UNQUALIFIED.
Den lesende Fold2 S/M5-baneaudit er COMPLETE/CONSUMED, exit0 under4G/512M.
På eksakt moden fitakse47338 og fast senere utvikling4096 hadde
etterpåkjent S-side negativ middel netto etter deklarert4bps rundturkost
ved alle forhåndsvalgte1/5/15/60/240/480min. Senere middel var
−5,419/−1,236/−1,230/−1,304/−0,412/−0,923bps; dette er ikke
modellvalgte handler. Ved480min var1209 senere sertifiserte S-side-rader
terminalobserverte, men bare986 hadde observert hel prisbane. MFE og
konservativ pre-MFE MAE er rapportert på den snevrere fellesmasken;
netto–MAE på akkurat samme rader er ennå ikke målt. M1 OHLC gir ikke
intraminuttrekkefølge. Kostautoriteten her har532bps årlig LONG-finansiering;
forrige femminuttersdiagnose deklarerte569bps, begge historiske antakelser.
Rapport: `docs/S_M5_PATH_COST_ALIGNMENT_RESULT_REVIEW_20261115.json`.
Original CLAIM/RESULT/TERMINAL:
`/home/andre2/GX1_RUNS/MARKET_STATE_S_M5_PATH_COST_ALIGNMENT_20261115_001/`.
0 modellforwards, optimizersteg, TEST, broker, handel eller Entry-valg.
S består som fem-TF markedstilstand; neste økonomiske bro er netto og
pre-MFE MAE på felles maske, ikke en ny handlingsregel.

Forrige policy: S_FOLD2_EXACT_PRIOR_LATER_COMPLETE_BASELINE_ONLY_EDGE_UNQUALIFIED.
Engangsauditen er COMPLETE/CONSUMED, exit0 under 4G/512M. Den frøs S/F-prior
på eksakt moden Fold2 TRAIN-akse (47338 rader) og målte proper-score på
4096 faste utviklingsrader fra jan–mar2025 fordelt på14 UTC-uker og alle fem
TF. Gammel fem-TF-prior brukte en annen 47602-raders fit-akse. M5
kjent-side-priorens Brier var0,248755 (S,1392 rader) og0,249257
(F,1591 rader). Dette er bare en frossen nullreferanse, ingen modellscore
eller økonomisk Edge. Resultat:
`docs/S_FOLD2_EXACT_PRIOR_LATER_RESULT_REVIEW_20261115.json`; original
CLAIM/FROZEN_PRIORS/RESULT/TERMINAL:
`/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_EXACT_PRIOR_LATER_4096_20261115_001/`.
0 modellforwards, optimizersteg, TEST, broker eller handel. Neste konkrete
gate er et separat kost- og prisbanebundet M5 Entry-mål med pre-MFE MAE.

Forrige policy: S_M5_QUOTE_REFERENCE_NEGATIVE_EDGE_UNQUALIFIED.
Fold2 TRAIN-only S/M5-quoteaudit er COMPLETE/CONSUMED, exit0 under 4G/512M.
Eksakt M1 bid/ask ved M5-beslutning og fem minutter senere var tilgjengelig
på47124 av47338 modne rader;16071 av16101 med sertifisert S-side. Selv med
etterpåkjent korrekt S-side var middel netto −0,6462bps etter bid/ask og
deklarert4bps rundtur-slippage;10033/16071 var ikke positive. Dette er en
måldiagnose på en selektert fasitkohort, ikke modellscore, realisert P&L eller
bevis mot en lærbar undergruppe eller annen horisont. MFE/MAE her mangler
rekkefølge før første MFE. S beholdes som fem-TF markedskontekst; separat
kost- og prisbanebundet M5 Entry-mål trengs. Aggregert rapport:
`docs/S_M5_QUOTE_ALIGNMENT_RESULT_REVIEW_20261115.json`. Originale
CLAIM/RESULT/TERMINAL ligger i
`/home/andre2/GX1_RUNS/MARKET_STATE_S_M5_QUOTE_ALIGNMENT_20261115_001/`. 0
modellfits, optimizersteg, TEST, broker eller handel.

Fold2 full-GC senere inputplan er COMPLETE/CONSUMED, exit0 etter169,04s
under12G/512M. Den gjenbrukte de seks fysiske sidecarene og bandt
4096 faste utviklingsrader fra jan–mar2025 i256 batcher til dagens
full-XAU+GC-visning:14 tensorflater, fem GC-flater, fem kausale
MTF-klokker på alle rader og finite råverdier.0 mål-arraylesing,
modellkonstruksjoner, forwards, optimizersteg, TEST og handel.
Eksakt ny dekningsbinding:
`configs/research/MARKET_STATE_FULL_GC_FOLD2_LATER_COVERAGE_20261115_001.json`.
Aggregert kontroll: `docs/S_FULL_GC_FOLD2_LATER_INPUT_RESULT_REVIEW_20261115.json`.
Dette kvalifiserer input på det faste panelet, ikke læring eller Edge.

Fold2 engangsplan002 er COMPLETE/CONSUMED, exit0 etter26,09s under
12G/512M. To faktiske TRAIN-rader gav full XAU+GC-featurebatch med
2727 lokale felt,255 MTF-felt på alle fem tidsrammer og rå GC-sekvenser.
Alle fem MTF-klokker var kausale. Valgte native S/F-mål, forward og
backward besto; fire rapporterte parametergrupper hadde finite og
ikke-null gradienter, maskerte ukjente/flate sider hadde null direkte
gradient. Modellhash var uendret;0 optimizersteg,0 TEST og ingen handel.
Dette er mekanikk, ikke dokumentert prediktiv nytte eller Edge.
Se `docs/S_FOLD2_SELECTED_NATIVE_INITIAL_RESULT_REVIEW_20261115.json`.

Den blokkerte engangsplanen 001 feilet lesende preflight før claim fordi dens
historiske v2-kildevisning ikke lenger matcher dagens runner og kilder. Den
skal ikke kjøres; outputroten finnes ikke.

Eksisterende fold2 TRAIN-kvitteringer har seks fysisk kvalifiserte
ankerspenn-sidecars, utvidet normalisering og bevart GC-historikk. Samme
kildeeiere binder nå en full-GC S/F-visning med2727 lokale signalfelt,
255 MTF-felt på M5/M15/H1/H4/D1 og eksakt historisk-til-nåværende GC-kildebro.
Syntetisk gather var bitlik full rå XAU+GC, og den fokuserte anchor-span-suiten
besto. Dette var kilde-/metadata-/syntetisk bevis før den fysisk beståtte
INITIAL-kontrollen ovenfor; ingen fit eller Edge. Fold1 avvises av uforenlig arvet GC-
enhetsprofil. Én separat test for relative GC-priskoordinater feiler på
en eksisterende gc_order_flow_v1.py-kildehash; rå-GC-ruten bruker den ikke.
Se docs/S_FULL_GC_ANCHOR_SPAN_FOLD2_SOURCE_REVIEW_20261115.json.

Den kronologiske valgte-native S/F-prioren og senere proper-score er målt
på alle fem TF. Direkte S-side som Entry-retning er nå negativ i middel
etter deklarerte kostnader ved seks faste observasjonshorisonter i fit og
senere utvikling. Før beslutningsstudie må terminal netto og pre-MFE MAE
samles på felles maske; før fit må én konkret kausal signalhypotese og
optimizer/resume kvalifiseres. Trening krever eksakt operatøradmisjon.
Begge nye engangsplaner er forbrukt. TEST, broker og handel er stengt.

Den valgte fem-TF S-masken er hashbundet til118249 modellrader og fem faste
native endepunkter. Den gamle samplede tidsforespørselen treffer D1s native
endepunkt sjelden: bare19 av6951 kjente D1-sidemål på fold1s TRAIN-prefiks
og37 av12291 på fold2s med eksakt pass0/seed20260911. De samplede native-
etikettene er bitlike med den valgte masken, men øvrige samplede tider er
fortsatt bare en kandidat. Eksisterende full-GC S/F-rute har allerede
side-normalisert tap og en liten fysisk optimizer-/resume-kontroll. Den
eldre S-only v2-batchen er bare brukt til nullsteg; den er ikke aktiv fit-rute.
Se `docs/S_MARKET_STATE_NATIVE_ENDPOINT_FIT_BINDING_REVIEW_20261115.json`.

Eksisterende S/F-batchadapter har nå en eksplisitt opt-in for valgte native
S-endepunkter. Den sender faste fem K til forward, leser valgte mål etterpå,
sjekker eksakt gyldighet og avviser umodne fit-mål; gammel samplet modus er
uendret.18 fokuserte batchtester og23 taps-/skårtester besto.
Kilde-/syntetisk rute og reelle målakser er nå fulgt av fysisk INITIAL,
men ingen fit, modellverdi eller Edge. Modusen er ennå ikke bundet i
recipe/campaign.

Fersk full-GC INITIAL er nå fysisk bestått på to TRAIN-rader. En egen
kronologisk full-GC S/F-fit med frossen tidligere-TRAIN-prior og proper
skår på fem native endepunkter er ikke låst eller innrømmet. Økonomisk
Edge, TEST og handel forblir stengt/ubevist.
H4/D1-vinduets kausale overlapp står i
`docs/S_MARKET_STATE_WINDOW_OVERLAP_REVIEW_20261115.json`.

Tidligere policy: TA_F_M5_PRICE_ATR_SOURCE_REPLAY_COMPLETE_EDGE_UNQUALIFIED.
Uavhengig capped lesereplay fra hashbundet M5-parquet rekonstruerte
118706/118706 F-M5-issues: neste lukkede M5-pris, Wilder ATR14 på siste
lukkede bar, eksakt endepunkt, gyldighet, tegn og normalisert størrelse
var bitlike med måltapen. Alle118249 modellraders sideetiketter var like;
0 numeriske avvik og0 modellfits. Se
`docs/TA_F_M5_PRICE_ATR_SOURCE_RECONSTRUCTION_20261114.json`.
Pris-/ATR-timing og målberegning forklarer ikke F-M5-overføringssvikten.
64-runders oppskrift og indre TRAIN-valgt nedkorting gav ingen positiv
senere Brier-gate. Ingen ny kilde-/modellfeil er målt som begrunner enda en
fit; behold kvitteringene og vent på en konkret kausal hypotese med
senere uavhengig evidens. TEST, native-trening og handel er stengt.

Tidligere policy: TA_F_M5_TARGET_ENDPOINT_GAP_REVIEW_COMPLETE_EDGE_UNQUALIFIED.
Lesende kontroll av den hashbundne F-M5-måltapen fant118249/118249
modellrader med eksakt framtidig5-minutters endepunkt,0 mellomliggende
gap og0 avvik mellom sign,0,5×ATR-størrelse og allowed-sideetikett.
Alle fem senere scorefoldene har0 mål-/endepunktsgap. Se
`docs/TA_F_M5_TARGET_ENDPOINT_GAP_REVIEW_20261114.json`.
Dette utelukker akkurat gap, endepunkt og sidebit-mapping som forklaring på
den målte F-M5-overføringssvikten; det gjør ikke målsemantikken prediktiv.
Ingen ny fit er innrømmet. Behold de negative kvitteringene; en ny F-M5-
hypotese krever et konkret kausalt kilde-/målbevis eller senere uavhengig
forwarddata. TEST, native-trening og handel forblir stengt.

Tidligere policy: TA_F_M5_INNER_TRAIN_TRANSFER_COMPLETE_NO_EDGE.
Indre F-M5 TRAIN-plan001 er COMPLETE/CONSUMED, exit0 etter163,19s under
12G/512M. Fem64-runders modeller fikk egen TRAIN-AUC0,7097–0,7974 og
bedre TRAIN-Brier enn TRAIN-prior, men alle fem tapte Brier mot prior på
senere indre TRAIN-validering. Alle fem eksakte splittene, priorene,
35 checkpoint-AUC/Brier og treningsmål er uavhengig gjenskapt fra lagrede
modeller; ingen ytre scorefasit ble brukt i fit eller rundevalget.
Den forhåndsbundne regelen valgte [0,0,0,2,1] trær. Lesende anvendelse på
gjenbrukte ytre utviklingsfolder gav Brier(prior−modell)=−0,00000447
på34760 F-kjente sider; paret uke-95%-intervall
[−0,00002630,+0,00001868] omfatter0. Bare fold4 bedret Brier svakt;
fold5 ble verre, og tre folder var kun prior. Dette beviser ingen stabil
rangering, kalibrering eller Edge og begrunner ingen ny ytre fit.
Se `docs/TA_F_M5_INNER_TRAIN_TRANSFER_RESULT_REVIEW_20261114.json`.
Neste steg er én konkret lesende F-mål-/kildeintegritetskontroll før noen
ny modellendring. TEST, native-trening, Entry/Exit, broker og handel er stengt.

Tidligere policy: TA_F_M5_INNER_TRAIN_TRANSFER_PREPARED_NO_FIT.
En egen F-M5 indre TRAIN-plan er PREPARED_NOT_ADMITTED uten claim eller fit:
`configs/research/MARKET_STATE_F_M5_INNER_TRAIN_TRANSFER_20261114_001.json`.
Fem modne prefiks er delt kronologisk ved80% av F-kjente TRAIN-sider;
inner-train-mål må bli gyldige strengt før valideringsstart, med eksakt
5-minutters F-horisont. Planen binder fem faste64-runders LightGBM-fits,
forhåndsvalgte1/2/4/8/16/32/64 readouts og TRAIN-prior fryst før fit.
Valg av runder kommer bare fra den indre TRAIN-valideringen; ingen ytre
scorefasit brukes i denne jobben. Neste steg er direkte og capped fysisk
preflight uten claim/datafit; først etter bestått kontroll kan egen
policyadmisjon åpne én capped kjøring. Ingen TEST, native-trening eller handel.

Tidligere policy: TA_F_M5_TREE_PATH_POSTHOC_DIAGNOSIS_COMPLETE_NO_FIT.
Lesende replay av lagrede F-M5-trær ved1/2/4/8/16/32/64 runder bekrefter at
64-runders Brier er dårligere enn fryst TRAIN-prior i alle fem senere folder.
Tidlige prefikser gir små positive Brier-forskjeller bare i fold1/2/4;
fold3/5 er negative ved alle undersøkte runder. Rangering forblir nær0,5.
Dette er post hoc på konsumert utviklingsscore og velger ingen runde.
Se `docs/TA_F_M5_SAVED_TREE_PATH_DIAGNOSIS_20261114.json`.
Neste avgrensede kontroll er indre kronologisk TRAIN-validering med eksakt
5-minutters F-målmodning under en egen forhåndsbundet engangsplan.
Ingen ny fit, TEST eller handel er åpnet.

Tidligere policy: TA_F_M5_TRAIN_SCORE_TRANSFER_GAP_CONFIRMED_NO_FIT.
Lesende capped diagnose av de fem lagrede64-runders F-M5-modellene gjenskapte
alle senere scoreprediksjoner eksakt fra den hashbundne4003-felts matrisen.
Eget TRAIN hadde AUC0,6984–0,7945 og Brier0,2280–0,2418, klart bedre enn
TRAIN-prior0,2498–0,2499. Senere AUC var0,5011–0,5135 og Brier dårligere
enn prior i alle fem folder. Dette bekrefter modellens TRAIN→senere
transfergap; det utpeker ikke én bestemt feature, treparameter eller
markedsregime som årsak. Rapport:
`docs/TA_F_M5_SAVED_MODEL_TRAIN_SCORE_DIAGNOSIS_20261114.json`.
Ingen ny fit eller kalibreringsparameter er valgt. Neste avgrensede spørsmål
er om en forhåndsbundet indre kronologisk TRAIN-kontroll med eksakt5-minutters
F-målmodning kan identifisere en rettbar overtilpasning. Ny fit krever egen
engangsplan under cap. TEST, native-trening og handel forblir stengt.

Tidligere policy: TA_F_M5_CURRENT_OOF_COMPLETE_NO_POSITIVE_GATE.
Den ene F-M5-tabellplan002 er COMPLETE/CONSUMED, exit0,187,61s under12G/512M.
Fem faste F-trente LightGBM-hoder brukte den eksakte4003-felts kausale visningen;
3991 gamle felt var bitlike og alle12 nye spennfelt var tilgjengelige før fit.
Alle fem egne TRAIN-priorer ble fryst før første fit. Lagrede prediksjoner,
F/S-mål, AUC, Brier, retrospektive strata og2000 parede uke-bootstrap-trekk
er uavhengig etterregnet. 34760 F-kjente senere utviklingsutfall gav
Brier(prior−modell)=−0,0009316; alle fem folder var negative, og95%-ukeintervallet
var[−0,0014261,−0,0004570]. AUC per fold var0,5011–0,5135. De nye
spennfeltene ble faktisk brukt i alle fem trærne. Se
`docs/TA_F_M5_CURRENT_OOF_RESULT_REVIEW_20261114.json` og originalkvitteringer
under `/home/andre2/GX1_RUNS/MARKET_STATE_F_M5_TABULAR_OOF_20261114_002`.
Den konkrete253/255-kildeblokkeringen er rettet; F-M5-rangering/kalibrering
har fortsatt ingen positiv gate. Ingen ny parameter er valgt fra scoreutfallene.
Planen relanseres ikke. En ny TRAIN-only kalibreringshypotese krever særskilt
forhåndsbundet plan og senere uavhengig evidens. TEST, native-trening,
Entry/Exit, broker, handel og økonomisk Edge forblir stengt/ubevist.

Lesende capped 4003-felts tabellvisning er nå fysisk kontrollert på
118249 modellrader. Prosjeksjon av alle3991 gamle felt hadde nøyaktig
samme SHA som frossen gammel matrise; de12 nye spennfeltene er hentet fra
kvalifiserte kausale sidecars. Ny matrise-SHA19539b3c. Ingen mål eller
modellfit ble brukt. Se `docs/TA_CURRENT_TABULAR_SPAN_VIEW_20261114.json`.
F-M5-OOF på denne visningen er målt ovenfor, uten positiv gate.

Fold3–5 har egne uforandrede253-felts TRAIN-normaliseringer. Kvalifiserte
fullsidecars har eksakte før-cutoff-prefikser på alle seks overflater.
Den eksisterende additive normaliseringskontrakten er utvidet med eksakte
fold3–5-parentbindinger; fold1/2-output forblir rekonstruerbart. En
én-gangs CPU-plan for bare36 nye spennstatistikker er COMPLETE/CONSUMED, exit0 etter8,55s. Tre nye foldspesifikke
255-felts TA-kontrakter er uavhengig rekonstruert;18 overflatefit gav36
spennstatistikker, mens alle gamle statistikker ble bevart. Fold1/2s
ferdige kontrakter gjenbrukes. Se
`docs/TA_FIVEFOLD_SPAN_NORMALIZATION_RESULT_REVIEW_20261114.json`.
Dette var inputnormalisering; den etterfølgende F-M5-OOF er målt ovenfor og har ingen positiv gate.

Plan002 er COMPLETE/CONSUMED, exit0 etter143,53s. Den bandt eksakt
2025-native-del og gjenbrukte bitlik local-delta fra konsumert plan001.
Seks fulle sidecars dekker290522 rader;173854 nye mot fold2, hvor71186
lokale rader ble gjenbrukt. Alle31 gamle trendlinjefelt hadde null bitavvik,
og uavhengig kontroll bekreftet bitlik gammel/ny sammenslåing, eksakte
rad-/tids-hasher og tilgjengelighet før TEST. Se
`docs/TA_FIVEFOLD_SPAN_EXTENSION_002_RESULT_REVIEW_20261114.json`. Lesende kildekontroll fant eksakt treff
M5/M15/H1/H4/D1 på71186/23735/5937/1552/258 nye rader.
`docs/TA_FIVEFOLD_SPAN_EXTENSION_001_FAILURE_REVIEW_20261114.json`
bevarer årsak og terminal. Ingen modell- eller normaliseringsfit.
Foldspesifikk TRAIN-normalisering og F-M5-OOF er siden fullført ovenfor;
kalibrering og økonomisk Edge er ikke påvist.

Femfolds manglende 255-felts input er nå bundet i en input-only plan uten claim:
`docs/TA_FIVEFOLD_ANCHOR_SPAN_COVERAGE_20261114.json` verifiserer30/30
opprinnelige fit-radhash og seks sammenhengende før-TEST-radintervaller.
Planen gjenbruker fold2s seks uforanderlige sidecars og lager bare173854
nye rader, samlet290522. Replay krever bitlikhet for31 gamle trendlinjefelt;
modell/normalisering/mål/TEST er utenfor jobben. Ny engangsplan
`configs/research/MARKET_STATE_TA_FIVEFOLD_SPAN_EXTENSION_20261114_001.json`
ble claimet én gang og feilet konsumert etter181,4s på M5-kildeklokken.
Local-deltaen på71186 rader besto31/31 gammel-feltparitet, men ingen
samlede sidecars ble fullført. Kilde/TEST/trening forble uendret. Gammel
native-delliste slutter i2024; eksakt2025-del finnes, og alle fem nye
native-aksene matcher cachens nødvendige rader. Plan001 relanseres aldri. Global trening er fortsatt
stengt; Edge er ikke kvalifisert.

Tidligere policy: TA_FOLD2_F_M5_BRIER_ACCOUNTING_COMPLETE_EDGE_UNQUALIFIED.
Gammelt F-M5 FINAL-hode hadde senere innen-UTC-dag AUC0,4724 mot
S-skårens0,5607 på samme F-mål; samlet F-TRAIN-AUC0,5125 skjulte
lokalt rangeringstap. Kilde- og etikettreview står i
`docs/TA_FOLD2_M5_SHARED_HEAD_SOURCE_REVIEW_20261114.json`.

Opt-in delt S/F-M5-sidehode er implementert uten endring i gammel
standardkontrakt, separate S/F-målmasker, aktivitet eller andre TF. Fersk
fysisk shared-INITIAL001 er COMPLETE/CONSUMED, exit0,508,64s,512
forwards og0 updates; bitlik S-output og F-aktivitet/andre TF mot gammel
INITIAL, ny F-M5-side bitlik gammel S-M5-side. Senere F-M5 nullstegs
AUC0,4868→0,5258, men Brier0,2504→0,2618. Utforskende paret dag/uke-
intervall for AUC-punktgevinst omfatter0. Se
`docs/TA_FOLD2_SHARED_M5_INITIAL_RESULT_REVIEW_20261114.json`.

Én særskilt hashbundet shared-M5 TRAIN-fit plan001 er nå fullført og
konsumert, exit0,4795,07s:47338 TRAIN-rader,2959 B16-besøk,2952
optimizersteg,7 skip,512 faste readout-forwards,0 checkpoint/TEST.
Kildecommit5db2040b, plan SHAdf869636, RESULT SHA4f03bac6,
terminal/vakt og alle130 readout-celler uavhengig kontrollert; ingen
kildeendring under kjøringen. Signed GPU-vakt exit0 med maksimalt58°C
kjerne,62°C minne,166,33W og2172MiB GPU-minne. F- og S-M5-logit
forblir bitlike på samme rad. Full gjennomgang står i
`docs/TA_FOLD2_SHARED_M5_FIT_RESULT_REVIEW_20261114.json`.

Senere F-M5 N1591 AUC gikk0,5258 INITIAL→0,5087 FINAL, Brier
0,2618→0,2500; fryst F-prior Brier0,24926 er fortsatt bedre.
Paret dag/uke-95%-intervall for AUC-endring−0,0171 omfatter0,
og begge FINAL-AUC-intervall omfatter0,5. Januar/februar/mars FINAL-AUC
0,5009/0,5215/0,5016. TRAIN F-M5 AUC0,4950→0,5020 og Brier
0,2580→0,24982; dette er TRAIN-fit, ikke generalisering. Mot gammelt
separat FINAL F-hode steg senere AUC0,4874→0,5087 og Brier ble lavere,
men sammenligningen er post hoc og paret intervall omfatter0. Modellens
senere gjennomsnittssannsynlighet0,4982 mot UP-andel0,5374 viser
fortsatt svak kalibrering. Ingen kvalifisert Edge eller økonomi.

Lesende kohortdiagnose på fulle TRAIN-mål:7556/19072 F-kjente M5-rader
mangler entydig S-side;7552 av disse har S allowed-set5={NED,OPP}.
På senere fast panel har F-only N636 mot S/F-overlapp N955. Shared FINAL
innen-UTC-dag AUC er0,4882 på F-only og0,5535 på overlapp; pooled
F-only AUC er under0,5 i januar/februar/mars. Post hoc dag-bootstrap for
innen-dag-gapet+0,0653 har95% intervall −0,0037 til0,1303, så kohortforskjellen
er usikker. Dette viser en mål-/dekningsforskjell, ikke en bevist kodefeil
eller grunn til å invertere/gate skåren. Se
`docs/TA_FOLD2_SHARED_M5_F_ONLY_DIAGNOSIS_20261114.json`.

Lesende femfolds mål-OOF er nå bundet på hele scoreaksen for fold1–5
(2024-10–2025-12). F-only utgjør38,86–41,71% av F-kjente M5-sider i
scorefoldene; nesten alle har S allowed-set5. Alle forkortede TRAIN-prefiks
har modne S/F-mål. Dette er mål- og dekningskontroll med0 modeller,
forwards og optimizersteg, ikke modell-OOF eller 26måneders kvalifikasjon.
F-only er definert av framtidig S-fasit og kan bare brukes som retrospektiv
scorestratum, aldri ONLINE-gate. Samme rapport forhåndsbinder femfolds
modellevaluering og viser at å legge fryst F-TRAIN-priorlogit+0,04719 til
shared FINAL gir senere Brier0,249257 mot prior-alene0,249257; ingen målt
verdi utover prior og ingen AUC-endring. Se
`docs/TA_M5_F_ONLY_FIVEFOLD_TARGET_AUDIT_20261114.json`.

Den forberedte femfolds F-M5-tabellplanen001 er REJECTED_PRECLAIM, uten
claim/output/modellfit. Eksakt import-/kvitteringspreflight besto, men
capped lesende fysisk inputvisning stoppet før matrisen; ny lesende
preclaim-vakt avviser også gammel tabellbredde: gammel bundet
26måneders MTF-cache har253 felt, gjeldende kilde krever255. De to
trendlinje-ankerspennfeltene kom inn12.11. Gammel tabellfeaturematrise
har3991 felt og ingen spennfelt; dagens modellkontrakt krever4003.
Ingen 255-felts cache ble funnet under GX1_DATA/GX1_RUNS. Den gamle
femfolds-normaliseringen fra07.11 er for før-spennprofilen; kun fold2s
nye255-felts native input/normalisering er særskilt kvalifisert. Ingen
nullutfylling eller stille dropping av genuine felt. Se
`docs/TA_F_M5_TABULAR_INPUT_BLOCKER_20261114.json`.

Eksisterende lagrede, S-trente tabell-OOF-prognoser ble scoret lesende mot
F-M5 på alle fem utviklingsfold: N34760, Brier0,251321 mot folds
frosne F-TRAIN-prior0,249755; alle fem folds er dårligere. Paret
uke-95%-intervall for prior-minus-modell er −0,002008 til−0,001096.
Dette er ikke en F-trent modelltest eller bevis for at F ikke kan læres.
Se `docs/TA_S_TABULAR_ON_F_M5_READONLY_REVIEW_20261114.json`.

Frosne fold2-shared-captures er nå dekomponert eksakt mot F-TRAIN-prioren
på fast senere panel (N1591). INITIAL→FINAL reduserte F-sannsynlighetenes
standardavvik ca0,0235→0,0021 mens AUC falt0,5258→0,5087. FINALs Brier
over prior er+0,000785; nivåskift-interaksjon+0,000694 dominerer,
scorevarians er+0,000004 og retningskovarians gir−0,000097 i kreditt.
Etter TRAIN-sentrering av den uendrede skåren er senere Brier0,249243
mot fryst prior0,249257, men punktgevinsten+0,000014 har dag-/uke-
95%-intervaller som omfatter0. Dette er utforskende på gjenbrukt VAL;
ren kalibrering endrer ikke AUC eller kvalifiserer Edge. Se
`docs/TA_FOLD2_F_M5_BRIER_ACCOUNTING_20261114.json`.

Neste steg er gyldige255-felts kausale inputs og foldspesifikk TRAIN-
normalisering for alle fem folds før en helt ny eksakt modellplan, eller
en særskilt bundet, smalere F-referanse. Plan001 kjøres ikke. Global
training_enabled/native_training_enabled/full_epoch_training_allowed er
false. TEST, Entry, Exit, broker, ordre, handel og promotion er stengt.

TA_FOLD2_M5_004_COMPLETE_CONSUMED_F_M5_EDGE_UNQUALIFIED. Edge-målet er
uoppnådd. TARGET_COUNTS001 er COMPLETE/CONSUMED, exit0,3,78s. RESULT ba50eb96,
VALIDATION dd45bd05, TERMINAL dccda6f6; ingen relansering. På eksakt47338 modne
TRAIN-rader /2959 B16 med siste10 er M5 kjent retning S N16101/q2890 og F N19072/q2950.
Sampled S-M5 er lik native M5-endepunktet; alle fit-targets er modne. Egne S/F-priorer
er fit på samme TRAIN-rader.9arkivdekodinger,2referanseprior-fit, null modeller,
forwards, feature-gather, optimizer, checkpoint, senere fit eller TEST.

Den eksisterende M5-equal-taps-/batch-eieren binder nå fold2s egne q/N, akser,
inputview og målkvitteringer; fold1-profil og default kontrakt er uendret. Forholdet
mellom fold1- og fold2-koeffisientene er1,00127(S) og1,00106(F), omtrent0,1%.
Dette er ikke en målt forklaring på svak ACC eller bevis for bedre AUC. En gammel
testhash ble korrigert mot uendret b248deda-baseline; den var foreldet før fold2-bindingen.
Fokuserte kildekontroller består. Fersk fold2 INITIAL er målt nedenfor.

Fersk fold2 INITIAL 004 er COMPLETE/CONSUMED, exit0,506,2s, plan6fe63d5e,
RESULT ebaa2de9, TERMINAL 7a44ec02, uavhengig VALIDATION 0c6abfbf.
512 B16-forwards på fast4096 TRAIN +4096 senere, én fersk modell, null checkpoint,
backward/optimizer/oppdatering/TEST. Modelltilstand uendret; signed GPU-vakt exit0.
Senere F M5 kjent-side N1591: AUC0,4868, ACC48,40%, alltid-UP53,74%, Brier0,2504.
Januar/februar/mars F M5 AUC0,4921/0,4980/0,4547. Dette er nullstegsprognoser
på gjenbrukt utvikling, ingen lært edge eller økonomisk kvalifikasjon. S H1/D1
AUC0,5612/0,5628 til tross for svak ACC; rangering og terskel-ACC skilles,
men tilfeldig initialisering er ikke bevis på nyttig modellrangering. Alle fem
TF og månedsceller er bevart i RESULT; 130 celler validert mot uavhengig AUC,
Brier og eksisterende ACC-eier. Plan002/003 er FAILED_CONSUMED før første forward,
med respektive normaliserings-/dualmodellhash-kontrollfeil; aldri relanser.
Plan001 hadde bare metadata-preflight, ingen claim.

Eksisterende trent oktober–desember-readout er gjenlest: oktober M5 AUC-gevinst
+0,0312, men paret dag- og ukeintervall omfatter0; november/desember FINAL AUC
0,5118/0,5051, samlet0,5078 med ukeintervall0,4844–0,5324. Oktober-ACC skjuler
lokal rangering, men ingen stabil M5-edge. D1-samlet AUC0,5499 har bredt
ukeintervall0,3881–0,7204 og er ikke M5 Entry-kvalifikasjon.
Fold2 M5-retning-alene-tapet har nå kilde-/gradientreview i
`docs/TA_FOLD2_INPUT_REVIEW.json`: kontrakt da042c0f, egen q/N, null direkte
gradient i aktivitet og andre TF. Én foreslått kronologisk pass bruker alle47338
TRAIN-rader/maks2959 B16-oppdateringer fra samme ferske INITIAL og skårer de samme
faste senere panelene, uten checkpoint-/terskelvalg. Tidligere fold1 fulltap ga
F M5 senere AUC0,5113→0,5101→0,5091 etter pass1/pass2 og dårligere Brier;
det er begrunnelse, ikke matchet fold2-kontroll. Eksisterende fit-runner importerer
gammelt checkpoint og binder fold1. Fold2 har fast B16/B10-kildekontrakt,
fit-kjerne og sju fokuserte syntetiske tester PASS, samt read-only metadata-
preflight og slutt-scorer. Scoreren replayet alle130 INITIAL-celler eksakt;
identisk INITIAL-par ga null AUC-gevinst. Bootstrap-seed er dag20260911,
mandagsuke20260912. Arbeidskoden binder alle17 uendrede INITIAL-inputkvitteringer;
design001/002 er fryst separat. Indre livssyklus er kildebundet og avviser
lukket/full-fit-svak policy og manglende GPU-vakt før payload. Den krever fersk startdigest,
fast pass og parvis FINAL-readout, men faktisk payload/modell/fit er ikke kjørt.
Fersk INITIAL-design er bitlik originalkilden og nå fryst i egen kildebundet
reseptfil; metadata-resepten binder 47338 TRAIN, maks2959 oppdateringer, null
checkpoint/resume og de 17 gjenbrukte inputkvitteringene. Campaignens
metadataautoritet kjenner fold2-resepten, 2959 besøkte batcher og mulige
hoppede oppdateringer i terminal cursor/progress. Materialisering, dispatch og
ytre worker er kildebundet og syntetisk rutet; syntetisk COMPLETE og midt-i-step FAILED publiserer
immutable resultat/cursor/progress, mens gjentak avvises. Plan001 bandt immutable kandidat-resept SHA4dac6a56 og Windows-boot649.
Den ble innrømmet og materialisert på ren kilde; fysisk boot650 startet
signert trainer. Runtime feilet før payload/modell/forward med RESULT SHAa331cbbc,
TERMINAL SHA2aab677b og kvittering SHA54ca0175:0 batchbesøk,0 updates.
Runtime leste toppnivå-`fold_id`, mens fryst resept har
`numerical_recipe.fold_id=2`.21 av22 kontrollerte adgangsfelt var sanne.
Plan001 er FAILED/CONSUMED uten relansering; Windows-oppgaven er deaktivert.
Eksisterende runtime-eier leser nå riktig schemafelt. Den gamle syntetiske
fixturen er rettet, og en negativ/positiv gatetest er lagt til. Design002
er bitlik design001 utenom ny engangsidentitet.13 fokuserte tester, syntaks,
lesende metadata-preflight og faktisk resept gjennom runtime-gaten før
payload består. Resept002 SHA31960926 er publisert fra ren commit0e365422, med målt
boot650 SHA94f48f81. Lesende preflight SHA27a5fecf og faktisk bundet
feature-/normaliserings-/initialidentitet SHAf21ee0dc er PASS; eksakt
resept når GPU-vakt før payload, og moduloppstart er røykprøvd uten claim.
Plan002 ble materialisert og startet etter fysisk boot651, men feilet med
RESULT SHA0babbcac, TERMINAL SHA2e4feb7b og kvittering SHAa41571f0:
0 batchbesøk,0 updates, ingen FINAL. Faktiske S/F-targeteiere har118249
rader, mens TRAIN-prefikset har47338; hele aksen ble feilaktig sammenlignet
med prefikset. Begge faktiske prefiks er eksakt like. Koden sjekker nå
prefikset, fullakse-regresjonen og14 fokuserte tester består; lesende
reell aksekontroll når første batch uten payload/update. Plan002 er
FAILED/CONSUMED uten relansering, Windows-oppgaven deaktivert. Design003
er identisk med002 utenom ny engangsidentitet. Resept003 SHAee16c4c4 er publisert fra ren commit853ddadc og boot651
SHA14a06a41. Lesende PREFLIGHT SHA27a5fecf og faktisk resept/
fullakse-gate SHA693ce1af passerer før forward/update. Plan003 ble materialisert og startet etter fysisk boot652, men feilet ved
første batch med RESULT SHA21204b71, TERMINAL SHAd8f2d199 og kvittering
SHA5ca23efb:0 batchbesøk,0 updates, ingen FINAL. Fit sendte endpoint-S-target
til evaluatoren som krever sampled S-target. Den latente bruken av manglende
report[count] ble også avdekket. Fit bruker nå sampled S i TRAIN og endpoint S
i fast readout, og evaluatoren gir eksplisitt M5-kjent-sideantall. Faktiske
S/F-targets og loss for første og siste batch passerer lesende, uten native
forward eller update. Plan003 er FAILED/CONSUMED uten relansering, Windows-
oppgaven deaktivert. Design004 er identisk med003 utenom engangsidentitet.
Resept004 SHAaa144f7f er publisert fra ren commit5f696c2e og boot652
SHA113db346. Lesende PREFLIGHT SHA27a5fecf og faktisk resept/
S/F-target-evaluator/loss SHA485ed0f2 passerer for første og siste batch
med 0 native forwards/updates. Plan004 fullførte én gang etter fysisk boot653: RESULT SHAd05f1226,
TERMINAL SHA1d1b4995 og validering SHAfa4bdf1a. 2959 batchbesøk,2952 updates,
7 skips,0 checkpoint,exit0; Windows-oppgaven er deaktivert. Fast senere
F M5 AUC var0,486777 INITIAL og0,487387 FINAL; parvis delta+0,000610
med dag95% [-0,033789,+0,033682] og ukeintervallet omfatter også0.
F M5 Brier ble dårligere0,250425→0,251524. Kun6/15 senere F TF/måned-celler
økte AUC og1/15 bedret Brier; alle celler er verifisert fra immutable
captures. S M5 fikk bedre punktestimat, men var ikke valgt primært F-mål.
Ingen kvalifisert rangering, uavhengig generalisering eller økonomisk edge.
training_enabled=false; delt S/F-M5-nullstegsbaseline er nå fullført og konsumert, mens egen fit-admisjon mangler. TEST, Entry, Exit, broker og ordre er stengt.
Edge er fortsatt ubevist. Lesende, hash-verifisert diagnose på uendrede captures
står i `docs/TA_FOLD2_M5_READ_ONLY_DIAGNOSIS_20261114.json`. Senere F M5
FINAL-sannsynlighet har middel0,4848 og SD0,00681 mot faktisk UP-andel0,5374;
AUC0,4874 og Brier0,25152. S/F-kjente M5-retninger er like på924/955
overlapprader, men FINAL-skårene korrelerer -0,530. S-skår mot samme F-mål
gir post hoc AUC0,5284 og Brier0,25007; dette er utforskende utviklingsdata,
ikke valgt signal eller edge. Kildekontroll fant samme native M5-representasjon,
separate sidehoder og korrekt UP-koding; ingen akse-/målfeil i captures.
Ren sannsynlighetskalibrering kan ikke forbedre AUC. Én ny kausal hypotese
er delt S/F-M5-sidehode mot begge modne TRAIN-mål; ulik etikett og F-only
dekning kan falsifisere den. Kildevarianten er nå særskilt dokumentert;
fersk baseline er nå målt, og ny engangsadgang kreves før eventuell fit.
Fokuserte fold2/campaign-tester PASS. Eldre fold1-fit-testfil har19 isolerte
fixture-feil NATIVE_F_PRIVATE_TA_M5_EXACT_INPUT_IDENTITY i uendrede kilder/refs;
ikke bruk den som fold2-kvittering eller gjenta uten konkret behov.

LATER_INPUTS002 er COMPLETE/CONSUMED, exit0,215,18s; terminal1297d154,
RESULT2cab8b60/COVERAGE5abd47c4/GATHER_RESULTe6aacfe9/VALIDATIONdf45d60f.
4096faste senere rader og alle åtte inputflater består.17353lokale rader gjenbrukt,
25028nye MTF,17944samlet gjenbrukte,42972fysiske rader,414bitlike overlapper.
Alle775868gamle MTF-celler bitlike.2025delen fjernet alle17343manglende M5stempler.
Dette rettet privat filliste, ingen påvist modellinputfeil eller prediktiv gevinst.
Full47338TRAIN-gather og89tester gjenbrukes. Aktivitetsfunksjon/egen normalisering/view
f7979c85 er uendret. TEST forseglet. Rapport: docs/TA_FOLD2_INPUT_REVIEW.json.

## Fullført og verifisert

SPAN_VIEW001: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001.
Kilde148a1505; RESULTa108db7b, exit0;744,71s.47338kronologiske TRAIN-rader,2959batcher
med siste10;23672normaliseringskall. Alle åtte inputflater har korrekt bredde/vindu/feltnavn,
alle rå native klokke-/alias-/finitetssjekker består, alle normaliserte verdier er endelige.
Besøkt foreldrerekkefølge matcher eget fold2-prefix. Null modeller/forwards/targets/fit/
checkpoints/oppdateringer.7,787milliarder skalare vinduspresentasjoner er overlappende
inputverdier, ikke selvstendige observasjoner. Kilde/policy/plan/backing-bookends PASS.
VALIDATION binder originalterminalens manglende TEST-felt til resultatets eksisterende false;
originalterminalen er bevart. Uavhengig JSON-/hashkontroll PASS. Ingen array-rerun.

Samme aktivitet-only-funksjon som FULL_PREFIX_ORDER: non-M5-side=False, uendret constructor,
ruting,317/132/255bredder,96lokale bars og16/64/96/96/252MTF-vinduer.51fokuserte tester,
CLI-preflight, materialisering og påkrevde commit-kontrakttester PASS. Fold1-rekonstruksjon
identisk. Fold2 har ingen enhetsunntak. Eksisterende GC bridge binder historisk ef64a40a
separat til dagens25881da8; ingen GC-input eller ny GC-kode. Mislykket frakoblet start kom
aldri til wrapper/CLAIM/data; samme uclaimede plan ble startet via vedvarende SSH-håndtak.
Denne kvalifikasjonen gjelder TRAIN-inputs; senere inputs/modellbaseline/prognoseverdi er ubevist.


SPAN_REUSE002: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002.
Kilde084d6c51; RESULT8f2acfaf, TERMINAL3de8d27a, VALIDATION9313ef0d. Exit0,264,72s.
116668rader=116197gjenbrukte+471nye. Alle seks rad-/tilgjengelighetsbevis matcher.
413overlapprader er bitlike for tid, native-posisjon og FP32spans. Tre gap: lokal181,
M5261, M1529; alle31gamle felt bitlike på nye rader (14601celler). Tre produsentkall,
15sidecar-dekodinger,6klokker; null targets/modeller/checkpoints/oppdateringer.
Rekonstruksjon fra kildestart beregnet2473843historiebars;471 er nye outputrader.
Eksplisitt subset er planbundet; tidligere seksflate-standard er uendret.11tester PASS.

SPAN_NORMALIZATION001: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_NORMALIZATION_20261113_001.
Kilde8541e352; RESULT372eaa86, TERMINALb0928b65. Exit0,3,56s under4G/512M.
Seks tofeltsfit på eget fold2-prefix; ingen gamle felt refittet. Alle6532gamle feltstatsett,
masker, kategorier, support, aliaser og fysisk lineage er bevart. Ingen fold1-enhetsunntak.
317signal/132kontekst/255perTF er projisert uten ny fit. Alle observerte spanverdier er
monotone og gjenfinnes eksakt etter heltallsavrunding; største inverse feil9,44e-6.
Basekontrakt7bd0f896; TA-kontrakt53e57991. TRAIN-gather er nå kvalifisert nedenfor; framtidig fit er ubevist.
27fokuserte tester, faktisk CLI-preflight og påkrevde commit-kontrakttester PASS.
Uavhengig kilde-/JSON-kontroll PASS. Fold1s beholdte normaliseringsartefakt er identisk.

Fold2 har47338TRAIN-beslutninger fra2024-05-01 til2024-12-30 23:50UTC. Tilgjengelighet
må være strengt før2024-12-31UTC; fysisk MTF-historikk omfatter også eldre vindusrader.
Original hashkontrollfeil og preflightfeil er bevart i001-root/policy. ROW_AXIS_CORRECTION
fastslår at rå aksehash var feil sammenligningsmetode; alle seks namespacede radakser stemmer.
Dette endret ingen accuracy-resultater eller rådata. Ikke gjenta denne metadataavklaringen.

## Hvorfor periodeoverføringen undersøkes

Siste prognoserapport: docs/TA_PERIOD_TRANSFER_REVIEW.json.
M5 F senere: accuracy51,21%, balansert50,39%, AUC0,507776; alltidUP51,92%.
UP-recall71,72%, DOWN-recall29,06%. Treklasse-argmax er alltidFLAT:58,76%, balansert1/3.
Kontrollert regneverk stemmer. M5 oktober AUC0,534610→0,565820 mens accuracy55,14→54,00%:
terskelaccuracy kan skjule bedre rangering i punktestimatet, men treningsgevinsten er usikker.
FINAL oktober M5/H1 AUC er over0,5 i begge punktvise dag-/ukeintervaller. Senere overføring
svikter i flere mål: H1 desember−oktober AUC−0,202292 utelukker0 i begge intervaller;
M5s månedsfall gjør ikke det. Brier forverres etter trening i november M5/M15/H1 og
 desember M5/M15/H1/H4. M15 november viser at én enkel gjennomsnittsbias ikke forklarer alt.
Alle fem TF og alle tre måneder er beholdt. H4/D1-unntak og usikkerhet står i rapporten.

PREPOOL_CAPTURE001/READOUT001 er ferdige; ikke gjenta dem. Se docs/TA_PREPOOL_PREDICTIVE_REVIEW.json.
Alle fem PREPOOL−SHARED AUC-/accuracy-intervaller inkluderer0; dette isolerer ikke all
informasjon eller pooling alene. Nære TF-rangeringer begrunner ingen ny arkitekturendring.

## Bevarte mål og ubeviste grenser

Alle genuine317/132/255features, fem TF, åtte familier og D1/makro beholdes. Tidligere
fullTA-konsum/paritet gjenbrukes; ingen ny feltcensus er nødvendig for disse to span-feltene.
COT2024-vintage forblir UNKNOWN; ingen konkret lekkasje er påvist. Juni2026 er gjenbrukt
utviklings-VAL. Alle tidligere statistiske intervaller er punktvise på gjenbrukt utvikling.
Tidligere fold1 INITIAL arvet eldre backbone-trening; fold2 INITIAL er fersk. Eksponerte rader blir ikke uavhengig OOF ved ny splitt.
Kjent-side-nevner er ikke valgbar Entry-dekning. Årsak, uavhengig generalisering, Entry/Exit-
kvalitet og økonomisk edge er ubevist. Korte forsøk dømmer ikke full historikk/kapasitet.
Full kvalifisert historikk/30–40epoker er senere retning, ikke en åpnet kjøring.
GC er på pause; M5/MTF/26månedersmålet består. TEST forseglet. Ingen broker, handel,
spending, promotion eller DATA/RUNS-opprydding.
