# CURRENT HANDOVER — XAUUSD, retning først

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`,
`gx1-3090-lan`. Les GX1_RULES helt ved første arbeid; start overtakelse med
`bash scripts/gx1_handover.sh --check`. NEXT_RUN_POLICY er eneste arbeidsstatus.
Root eier integrasjon, policy og launch. Historiske detaljer finnes i Git og
originale eksterne kvitteringer; ikke åpne dem uten konkret behov.

## Mål og aktuell status

Bruker spisset24.10 målet til **prisretning først**: bedre opp/ned-prognoser og
etterprøvbare sannsynligheter. Selve trades, MFE/MAE-optimering, risiko og Exit
utsettes. Målet er aktivt og ikke oppnådd. Teknisk konsum, TRAIN-fit,
generalisering og økonomisk edge er forskjellige bevis.

Siste målte retningsreferanse249 er COMPLETE/consumed og forkastet;actualpeer
er PASS. Også247 er forkastet/peerPASS. Datagrunnlag245 er kvalifisert.
Klokkediagnose246 er COMPLETE/consumed og actualpeerPASS,men kvalifiserer
ingen hel bookpakke.

**249 er COMPLETE/consumed; kandidaten er forkastet.** Åtte syntetiske
adaptercases og ekte current/predecessor-kvalifikasjon bestod før30CPU-fits.
251 låser én fast ExtraTrees-retningstest med alle3345 kvalifiserte snapshotfelt,
samme femTRAIN-folds og renprisfasit245.252 eksternworker er fryst0444 med
uavhengig feature-/estimatorreview og root-PASS.24hjelpere er AST-identiske
med249;50kildeeiere og firedatarefs er bundet. Nøyaktig én ny cappedCPU-fit
innrømmes, først seks score/datagrupper og én separat syntetiskforest, deretter
15genuineforests. Globaltraining/local252=true kun for dette avgrensede
forsøket; native/full_epoch/fullVAL/TEST/Exit/trades=false. Ingen faktisk
kvalifikasjon eller prognoseforbedring er ennå målt. Root lukker fitgater etter
terminal og krever actualpeer førresultattolkning. Prospektiv4GiB/1800s;
faktisk ET-tids-/minnebruk er fortsatt umålt.

## Hva faktisk er målt om retning

Native232 med218GC-history/support-pakken hadde7391updates og32finalforwards
på256TRAIN+256CONTROL, alle480horisonter. Teknisk guard og actualpeerPASS;
kvitteringer/checkpoints er originale. Den forbedret ikke Entry-retning:
CONTROL1/5/15 retnings-MSE mot191 var +0,647% /−0,291% /−0,185%; bare3/6
primærpunkter mot egeninitial og191 bedret seg. Alle12 dag/uke95%-CI krysset0.
Alle512 aktuelle handlingsrader varFLAT. TidligMFE/preMAE kunne bedre egeninitial
uten bedre retning; mot191 var alle8CONTROL-sidepanelene verre.

236s faste femfold snapshot/ridge-studie fullførte40fit/120regresjoner,
8armer og413667kodede foreldre. Ingen av sju familie+makro-armer bestod låst
videreføring. Alle14poolede5/15min-paneler var verre enn makro og prefixmiddel.
Dette er en avgrenset lineær snapshotkonklusjon, ikke dom over allefeatures,
lange sekvenser eller ikke-lineær informasjon. Ingen vinner eller ny fit følger.

247 er den første rene prisretning/logistikkreferansen, S_PRICE66 med61makro,
4session og ret_1. Fem ekspanderendeTRAIN-folds, fast239L2/C1/lbfgs/maxiter500,
15fit/30prediktorpopulasjoner;867faktiske solveriterasjoner(24–107),alle15success.
8,126sworker/8,423sytre; fryst fefd62d3/policy,7SHA-verifiserte inputs.
0normaliseringsfit/nativeF/U/CONTROL/TEST. Actualpeer avstemte8JSON,
9payloadfiler medstat,33paneler,231forvekslingsmatriser og132lagrede intervaller.

| Horisont | Binære OOF-rader | Modelltreff | AlltidOPP | Balansert treff |
|---|---:|---:|---:|---:|
|1min|88102|50,0624%|50,3099%|50,0101%|
|5min|87944|49,9909%|51,0086%|49,8935%|
|15min|87344|49,8443%|51,6899%|49,7829%|

Alle trepooledlogloss/Brier var verre enn egen modneprefixprior. Alle12pooled
proper-score dag/uke95%-CI lå helt over0(model−prior), ikke bare over null i
punktestimatet. Bare1/15foldpanel bedret beggepropermetrics. Ingen horisont
bestod fastfollowup. Dette forkaster S_PRICE66-fastlogistikk på gjenbruktTRAIN;
ingen generell feature-, native-, generaliserings- eller edgekonklusjon.
Ingen247relaunch/HPO/terskelvalg/kalibreringsfit.

## Kvalifiserte data og eiere som skal gjenbrukes

239s tre nye analysehjelpere er kildekvalifisert:50/50syntetiske casesPASS,
0fail/error/skip,eksakt1syntetisk sklearnfit og30små surfacefits. Renpris-,
temporal- og sannsynlighetseiere er uendret. Ingen ny fullsuite er begrunnet.

245 publiserte ren F64-midtprisfasit ved D=bar_start+300s og T=D+1/5/15min,
fra eksakte M1-openklokker. Full9-array198kostnads-/klokkeparitet er bit-eksakt;
originale targets/kostnader er bevart. Alle118249TRAINforeldre beholdes.
Observerte nullbevegelser1/5/15=476/194/99; manglende=12/450/1313.
Gammel kostnadskontrast hadde annet fortegn på503/212/109observerte rader.
Dette er målkorrigering, ikke dokumentert forklaring på manglende edge.

245 COMPLETE/consumed:235,084sworker/235,696sytre;116endpoints/116converts/
1TRAINreader/2eksisterendeh1guards. Ingen nye h5/h15pathquery,CONTROL-targetquery,
normaliseringsfit,prediktorfit eller nativeF/U. Quoteindeksen ble frigjort før
featurekoding. Fem S_PRICE66-matriser koder413667foreldre;109208088råfeaturebytes.
Sju readonlypayloads130260690B. Rchar24734894825B<32GiB,prosesspeak3170807808B.
ActualpeerPASS påmetadata/stat; peer åpnet ikke opaque payloadbodies.
Canonical191-scope omfatter også CONTROL-integritet; ikke påstå0CONTROLbytes.

Gjenbruk206TRAINraw, saved204SOURCE_MAP og209full-layoutbro til210. Fem210
baseline-prefixoppskrifter er allerede fit og uendret:20241001/20250101/20250401/
20250701/20251001. Ikke bygg historisk kildekontrakt på nytt og ikke refit.
245featureorden er155..215,3341..3344,1; sistekolonne65 er normalisert ret_1.
Persistens bruker derimot rå206ret_1-sign, med rånull som avståelse.

## Målt klokkeforskjell og M5-rute

246 COMPLETE/consumed:82,598sworker/83,102sytre,fryst2d37c138/policy;
5TF/175aggregerte paneler,0target/fit/nativeF/U. Rchar7494014023B<8GiB.
ProsesspeakRSS4348985344B kan inkludere delt mmap og er ikke cgrouppeak.
ActualpeerPASS kontrollerte kvitteringskjede,14kilde-SHA og alle175aggregater.

|TF|Verdi-bitavvik|Maske-bitavvik|
|---|---:|---:|
|M5|0|0|
|M15|853|573|
|H1|4701|846|
|H4|3861|3250|
|D1|24285|20465|

Alle avvik ligger der206s GCkalender-asof er nyere enn siste nativeprisclose
brukt av223. Samme klokke har0bitavvik i alle5TF. M5current er bit-eksakt på
alle118249foreldre. Dette forklarer242porten;239full5TF-paritetsport er uendret,
og hele86-kandidaten er fortsatt ukvalifisert. Ingen prognosenytte er vist.

223s tre M5-sidefiler er readonly,349969852B samlet: featuresF32[136069,639],
native_row_indicesI64[136069],available_nsI64[136069]. Kolonner502/503 er genuin
observed_book_tw_imbalance og observedmaske, råverdier med fast[-1,1]-skala.
M5current hadde117628observerte(inklusive19genuine nuller) og621maske0.
223s heleTRAIN-normalisering skal ikke brukes som OOF-prefixnormalisering.

## Låst og fullført hypotese:248/249

248DESIGN SHA305f3a29… og DESIGN_SOURCE_REVIEW00fbded1… låser to småarmer:
R1=245kolonne65/ret_1 alene; R1_M5_BOOK5 legger til current og previousM5
imbalance+mask. Bredder1/5,senere eksakt30fit/60prediktorpopulasjoner.
R1 er kontroll; kun augmented er kandidaten. Ingen feature-/horizont-/modellvinner,
HPO, terskelvalg, kalibreringsfit eller ny normaliseringsfit.

249s eneste kildeeier er learning_audit for ekstern
`NATIVE_M5_DIRECTION_PRICE_BOOK_REFERENCE_20261024_249/fit_price_book_direction_guarded.py`.
Root eier PLAN/review/policy/launch, feature_audit kildepeer. Gjenbruk239learner
og247IO/evaluering. Underliggende245/244metadataidentitet må holdes separat fra
ny248design. Ti eksisterendepayloads480230542B; ingen ny cache/206/quote/GCparquet.

Før fit må ny M5-adapter bevise currentexact available==D og identiske246
canonical currentID/start/available/pairhashes.223 er sparselagring: previous er
lookup av eksakt nativeID−1, aldri forrige sparselagerrad eller D−5min.
CurrentID0 er ukjent predecessor medverdi0/maske0/ID−1 uten negativindeksering;
positivcurrent med manglendeID−1 stopper. Krev prevclock<currentclock==D,
bevar reelle gap og alleforeldre, mål støtte/gap og bind ID-/klokke-/parhashes.
Bookverdier må være finite/[-1,1],masker binære ogmaske0⇒verdi0; ingenclipping.
Åtte småsyntetiske adaptercases skal bestå før genuineinput. Publiser faktisk
readonlyFEATURE_QUALIFICATION før første fit, medpositive modneprefixstøtter.

Samme245targets/femfolds/helehorisontmodning og alle null/missingnevner gjelder.
Augmented må slå egenprefixprior OG samme-radR1 påpooledlogloss/Brier; begge
propermetrics dag/uke95%upper<0 motbegge, minst4sammefolds forbedrer begge
metrics motbegge referanser, påalle1/5/15. Krev beggeprediksjonssider,
balanserttreff>.5 og treff overprior-majoritet/alltidOPP/råpersistenscommoncohort.
UTCdag/Mandaguke2000bootstrap/seed20260911/min8, fortsatt gjenbrukt utvikling.
Korrekte kontrastnavn; ikke arve247s prior-navn påaugmented−R1.

Prospektiv249ramme:én audit300+10s/4GiB/512MiB/64pids/éntråd/3GiBrchar,
64MiBnumpyoutputs. Rammen er consumed etter én faktisk kjøring. Ingen relaunch eller ekstrafit følger.

249 COMPLETE:7,175sworker/7,493sytre,fryst0855d6d2/policy.8/8adaptercasesPASS
førgenuinepayload,10inputSHA/8unikeNPYheadere/15prefixstøtter og1publisert
faktiskfeaturekvalifikasjon førfit. Current/previous observerte117628/117627;
alle118249 har eksakt nativeID−1. Reellelagsgap bevares;min300s/max263100s.
30fit/60predictor,189solveriterasjoner(2–12),0normalisering/nativeF/U/CONTROL/TEST.
Rchar574946057B,prosesspeak528715776B,readonlynumpyoutputs16625994B.

Bookkandidat1/5/15treff49,6833/50,9495/51,6280%,underalwaysOPP
50,3099/51,0086/51,6899%. Alle12dag/ukeproper-loss-CI motR1 krysser0;
booktillegget har ingen påvist stabil merverdi i denne lille lineære modellen.
Fellesnegativefolds1/1/1;allefollowupfalse. R1-kontrollens15minproper-scores
bedrerprior nominelt i dag/uke,menR1 blir ingenautomatiskvinner ellertrade.
Dette avgjør ikke fulle ikke-lineære/temporale features. Actual249peerPASS:9JSON/12payloadstat/66paneler/462confusions/396CI.
250 er fullført som kilde-feasibility.251 låser og tre sourcepeers har reviewet
én fast ExtraTrees(32trær/dybde8/minleaf256/sqrt/éntråd/seed20260911) med
alle19parametre.252 eksternworker er kildekvalifisert/fryst, learning_audit er eneeier.
Root eier policy/PLAN/launch; andrepeers leser. Én ny cappedCPU-kjøring er
innrømmet; ingen native/full_epoch/fullVAL/TEST/tradeautoritet. F32énfold/énhorisont, rawmap og encodedmap lukkes førfit,
fit_X slippes førto heleprediksjonspopulasjoner. Prospektiv4GiB/512MiB,
64pids/éntråd/1800s/8GiBrchar/2GiBscratch; faktisk ET-tid/peak er umålt.
Førgenuine skal seks score/datagrupper og én separat syntetiskforest bestå.
Genuinekontrakt:15forests/480trær/30populasjoner, femfoldproofs og15egne
modellfiler med verifisertstate, uten ekstra prediksjoner/refit. Frosset249R1
brukes barepåFOLD/pooled; prefixkunprior. Råp tilBrier/reliability/p>.5,
F64epsklipp kunlogloss. Førhverfoldfit måfullcol1ret_1 bitmatches249col65.
Alle1/5/15 må slåprior+R1 etterlåstgate; ingenHPO/terskel/kalibreringsfit/
automatiskvinner. Fulltemporalsekvens og parkertchart er fortsatt umålt.

## Varige grenser og ubeviste forhold

Native åtte spesialister ogniendeD1/makro, genuinefeatures/tidsrammer,
055/057/111 og eksisterende vakter beholdes.19.10kompaktprofil har eksplisitt
7aktivefamilier;chart er vedtattparkert,ikke utilsiktet fjernet. NativeGC-rutene
nårencoder/curvehead/backward/AdamW; ekte per-featuregradient-/parametergevinst
og generell temporal/ikke-lineær featureverdi er ikke bevist. Den nye lille
forskningsreferansen erstatter eller fjerner ingen nativefeature.

Siste nativefunksjon beholder223GCinputs/227freshstate/229egeninitial/230stateaudit/
231privatequalification/232epoke/233analyse. ONLINE-endring krever nyinitial,
men ingen slikendring er gjort. Originale kvitteringer/checkpoints bevares.
237 er stoppet førdata etter brukerens retningsprioritering;240/241/242/243 er
failed/consumed med bevartoriginalfeil;239/245/246/247/249 ercomplete/consumed.
Ingen gammelplan relanseres. Tidligere tekniskePASS skal ikke gjentas rutinemessig.

Fysisk grunnlag er23 gjenbrukte utviklingsmåneder:TRAIN mai2024–desember2025,
CONTROL januar–mars2026.26måneder er målet; april–juni er ikke fysiskbundet.
Juni er gjenbruktutviklingsVAL. TEST frajuli er forseglet for alltilpasning/utvalg;
bare vedtattmånedligforwardjournal scorer utenrefit. IngenExit/broker/live/paper/
ordre/spending/promotion.1–480min er beregningshorisonter,ingen taps-/holderegel.
Lang støtte bruker alltidLONG/kjøp-og-hold somreferanse. Økonomi må ta med alle
valgtehandler og åpneposisjoner; ingen slik handelsøkonomi er målt her.

Én tungCURRENT-jobb gjennom gx1_capped_run.sh med faktisklås/hardwarevakter;
fryst kilde/policy til terminal. Stabil langkjøring ses omtrent15–30min eller
sjeldnere,ingenminuttpolling. IngenDATA/RUNS-rydding uten retention-eier.
Parameterstate-audit ognyttretningsedgebevis er ikke startet/oppnådd. Oppdater
policy/handover i sammebølge; avslutt medfokusertverifisering/AST/stale-path/
gitdiff--check ogærlige ubevistegrenser. Pushrett tilwork/gx1-current gjelder
kode/docs/interne refs/aggregater,aldri rådata/vekter/hemmeligheter ellerforcepush.
