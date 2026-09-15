# Gjeldende status — Entry lærer selektive valg, 2026-09-16

Native fortsettelse99→115 er ferdig på frossen kilde b493e585.1024 nye
optimizersteg tok2679.503783s (44m39.5s) inkludert oppstart/observer, uten reboot.
Checkpoint115/global7057/epoch1offset2976 er diagnostisk. GuardPASS, trainer0,
observer0; Windows-task Disabled22:44:31UTC2026-09-15. Ingen aktiv tung jobb.
Planen er uttømt. Original95/99, lagret epoch_order og fast lærer91 er bevart.
Alle726 Adamtilstander økte1024; EMA25941→26965, offset19908.236tester og normal
commit-hook bestod før kjøringen. Modell, tap og gradientgrenser er uendret.

Paret CPU-måling99→115 er ferdig, session34866/exit0,27.498018s. Målte99outputs
ble gjenbrukt og deres samlede statistikk matchet nøyaktig; kun115 ble forwardet.
Samme64cachedinputs/targets/dropoutfrø; ingen korpus-/lærerberegning, optimizer,
backward, GPU, VAL eller TEST. En første evaluatorvariant hadde variabelkollisjon
før modellkonstruksjon; kun måleskriptet ble rettet. Feilforsøket er bevart.

Entry-Q-MSE17.321052→7.085304, ned59.09%. Valgene endret seg fra64FLAT til
61FLAT/2LONG/1SHORT. Alle tre handelsvalg stemmer med lærerens foretrukne side;
55av55 lærer-FLAT forblirFLAT. Seks av ni lærerforetrukne handler overses fortsatt.
Exit-MSE0.342115→0.055011. HOLD/EXIT-treff mot mål ca94.92% LONG og95.31% SHORT.
Forecast-retning ved5/25/60/120 nominelle minutter:78.125/81.25/84.375/89.0625%.
Dette er lokal læring etter samlet1280steg/320repetisjoner av64 TRAIN-innganger.
Ingen generalisering, sannsynlighetskalibrering eller lønnsomhet er dokumentert.

Avslutt nå fixed64-øvelsen; ikke jag perfekt treff på disse ni lærerpreferansene.
Neste arbeid er en avgrenset native fortsettelse på virkelige ettårs-TRAIN-rader
fra bevart IKKE-replay checkpoint95/global5777/epoch1offset1696, frem til neste
fulle juni utviklings-VAL. Gjenbruk modell/tap/profil og eksisterende tekniske
bevis.115 skal ikke promoteres til produksjon eller telles som årsdekning.
Avklar bare nødvendig resume-/sourcebinding i eksisterende native eier; ingen
nye modeller, terskler, tapsendringer eller ekstra samme-kohort-prober nå.
Ingen ny kjøring er ennå forberedt eller startet. Vurder faktisk policyøkonomi
etter neste VAL før enda en epoch. TEST forblir forseglet.

Bevis: handover_snapshot/ENTRY_REPLAY_CONTINUATION_REVIEW_20260916.json,
ENTRY_REPLAY_CONTINUATION_PAIRED_RESULT_20260916.json og native receipt.
Operatører/logger/resultater er bevart under NATIVE_ENTRY_REPLAY_CONT_B493E585.
RUNNING_NATIVE_CALIBRATION.json inneholder full binding og status.

# Gjeldende neste arbeid — fortsett samme læringskontroll fra99

Ny avgrenset CPU-observasjon av beholdt checkpoint98 er ferdig, session1529/exit0,
7.9742s. Samme64cachedTRAIN-inputs og samme diagnostiske dropout. Entry-MSE
95:19.576542 →98:17.918624 →99:17.321052; alle64 fortsattFLAT. Siste64 oppdateringer
reduserte feilen0.597572. Dette støtter videre avgrenset innlæring; ingen påvist
konvergens/platå. Ingen tidligere forwards/evalueringer ble gjentatt. Operatør og
logg er bevart under den avsluttede64-kontrollen; resultat i handover_snapshot.

Hypotesen om L1-median kontra forventningsverdi er ikke bekreftet som årsak.
Ikke endre forecast-tap, gradientgrenser eller modell nå. Fortsett nøyaktig samme
kombinerte læring fra99/global6033 til høyst7057:1024 ekstra optimizersteg på
samme64innganger,256 videre repetisjoner. Frossen lærer91, original epoch_order,
alle checkpoints og modell-/optimizer-/EMA-/RNG-tilstand ved overgang bevares.
Dette er ingen ny årsdekning; diagnostiske checkpoints er ikke produksjonsresume.

236 målrettede tester består. Bare tre eksisterende eiere for kjøregrense,
loader/resume og EMA-historikk er justert. AST-verifikasjon viser seks numeriske
forward/loss/backward/optimizer-eiere uendret. Ingen refaktorering eller ny modell.
Neste: commit/push, én faktiskCPU-overføring av99, deretter én native guarded
kjøring fra frisk fysisk boot. Ingen aktiv tung jobb ved denne commit. Gjenbruk
cachet99-resultat for paret99→115 etter kjøringen; hele epocher/VAL/TEST sperret.
Se RUNNING_NATIVE_CALIBRATION.json og NEXT_RUN_POLICY.json.

# Gjeldende status — lærbarhetskontroll ferdig, 2026-09-15

Frossen målekilde var GX1_CURRENT/work/gx1-current, commit6c9328d6.
Én native kontroll gjentok de samme64 faktiske TRAIN-inngangene64ganger med
uendret kombinert trening og fast lærer91. Checkpoint95/global5777 ble bevart;
privat checkpoint99/global6033/epoch1offset1952 er diagnostisk og skal ikke
fortsette produksjonstrening. Original epoch_order er uendret; ingen ny årsdekning.

Native kjøring tok1199.510192s inkludert oppstart/observer, uten fysisk reboot.
GuardPASS, trainer0, observer0, Windows-task Disabled21:17:36UTC; prosessene
764/709/658 er terminale. Planen er uttømt. Ingen tung jobb er aktiv.
216 målrettede tester og normal commit-hook bestod før kjøringen. Faktisk
CPU-overføring bevarte alle15 komponenter. Alle726 Adamtilstander økte256;
EMA25685→25941, historikkoffset19908 og lærer91 beholdt. Dette er ikke en ny
kontinuerlig-mot-delt-resume-verifikasjon; tidligere relevante bevis gjenbrukes.

Paret CPU-evaluering95→99 er ferdig (session57439/exit0,41.761691s,
peak2266684KiB). Fire opprinnelige95-referanser ble gjenskapt bitidentisk;
begge modeller brukte samme cached inputs/targets og diagnostiske dropoutfrø.
Ingen ny korpus-/lærerberegning, optimizer, backward, GPU, VAL eller TEST i evalueringen.

Målt på det gjentatte TRAIN-utvalget:
- Exit-MSE6.888519→0.342115: ned95.0335%. HOLD/EXIT stemmer med lærerens
  unike preferanse i88.6719% LONG- og90.2344% SHORT-tilfeller.
- Forecast-retning ved nominelle5/25/60/120min:56.25/60.94/54.69/54.69%
  →67.19/70.31/84.38/89.06%. Dette er L1-returanslag, ikke kalibrerte sannsynligheter.
- Entry-Q-MSE19.576542→17.321052: ned11.5214%, men fortsatt64FLAT mot
  lærer55FLAT/3LONG/6SHORT. Ingen av de ni lærerforetrukne handlene velges.

Dette viser lokal læring i Exit og markedsprognosene; Entry-handlingenes
økonomiske skille henger etter. Ingen generalisering eller lønnsomhet er bevist.
Entry-mikser/head endrer vekter, LR er9.890738e-5 og Entry-tapsvekten er
2.02215→1.95647, altså ikke slått av. Koden stopper fortsatt Entry-Q-gradienten
før markedsrepresentasjonen. Forecast lærer M5-close-retur over5–120min;
Entry-verdien inkluderer første M1-lukking pluss svak videreverdi fra lærer91.
Ulike fortegn mellom disse målene beviser ikke tidsfeil.30av256 observerte
Bellman-overganger er state0; ikke påstå at første tilstand aldri trenes.
Ingen lagret native tapskurve finnes i95/98/99; ikke påstå platå/konvergens.

Neste prioritet er å isolere Entry-representasjonens læring av økonomimålet,
med selvstendig markedssignal og fortsatt skjerming mot Exit-bootstrap.
Avgrens én presis target-/gradienthypotese før ny kodeendring; gjenbruk denne
kontrollen og eksisterende native vakter. Ikke innfør terskler/klassevekter,
bytt modell, gjenta fullførte prober eller start en hel epoch på dette grunnlaget.
Ingen ny produksjonsendring eller neste treningsplan er bestemt.

Bevis: handover_snapshot/ENTRY_LEARNABILITY_CONTROL_REVIEW_20260915.json,
ENTRY_LEARNABILITY_PAIRED_RESULT_20260915.json og tilhørende native receipt,
checkpoint-/forecastgjennomgang. Operatører og logger er bevart under
NATIVE_ENTRY_LEARNABILITY_6C9328D6. RUNNING_NATIVE_CALIBRATION.json er operativ status.

## Avsluttet FQI-kontroll og korrekt TRAIN-måling, 2026-09-15

Eneste kilde er /home/andre2/src/GX1_CURRENT, work/gx1-current. Frossen kilde for
fullførte målinger var f11c1dbeef12721bef58449dfac37fd735f4694f. Checkpoint95 har
global5777, epoch_index1/offset1696. Native256 steg tok1194.683976s inkludert
oppstart/observer, uten fysisk reboot. GuardPASS, trainer0, observer0. Windows-
task er Disabled; ingen aktiv tung jobb. Planen er brukt opp. Ingen ny lærer-
oppdatering eller hel epoch starter automatisk. TEST forblir forseglet.

169 målrettede regresjoner og vanlig commit-hook bestod. Faktisk CPU-overføring
kopierte target nøyaktig fra ONLINE91 og bevarte alle14 øvrige komponenter.
Native beholdt denne targeten gjennom256 steg. Alle726 Adamtilstander økte256;
EMA25429→25685, offset19908 beholdt. Ingen ny kontinuerlig-mot-delt-resume-påstand.
Gammel lærer og alle resultater/checkpoints er bevart. Kun target ble endret;
økonomiske formler, MSE, optimizer, modell og risiko er uendret. Dette er én
kontrollert læreroppdatering, ingen innført permanent «hver256»-regel.

## Resultat på faktiske native TRAIN-batcher

Korrigert paret CPU/no_grad-måling er ferdig, session3258/exit0,419.149976s,
peak9291828KiB innen eksisterende20GiB/512MiB-cap. Fire forhåndsvalgte faktiske
batcher fra kontrollen: epoch1offset1440/1525/1610/1695.64entries overalle12måneder,
4–7 per måned, og256 gyldige HOLD-celler per side. Lagret epoch_order og ordnet
materialisert sampleplan matcher nøyaktig. Samme nye lærer91, inputs, targets
og diagnostiske dropoutfrø før91/etter95. Ingen historisk GPU-dropoutreplay-påstand.
Alle source/checkpoints/inputs/targets/policy/RNG er bevart. Ingen nytrening,
backward, optimizer, VAL eller TEST ble brukt til målingen.

HOLD-MSE LONG14.09770→13.96113, nullbaseline14.10036; SHORT13.87834→13.59294,
nullbaseline13.51526. Samlet MSE falt1.5083%, men er bare0.2229% bedre enn null-
baseline. LONG-HOLD25→199/256; SHORT-HOLD256→205/256. Dette er forbedret Exit-fit
på utvalget, ikke bevist forventningsverdi, generalisering eller lønnsomhet.
Entry-Q-MSE19.43136→19.57654; studenten64FLAT begge, læreren55FLAT/3LONG/6SHORT.
Prognose-MAE bedres svakt på nominelle5/25/120min og forverres på60min; retning
bedres bare på5min. Prognoser er L1-returanslag før kostnader, ingen kalibrerte
sannsynligheter. De64 radene ble sett i den avsluttede treningskontrollen.
Ingen matchet kontroll med uendret lærer fra91 ble kjørt, så årsakseffekten av
selve læreroppdateringen er ikke isolert.

Se handover_snapshot/ACTUAL_WINDOW_PAIRED_TRAIN_RESULT_20260915.json og
ACTUAL_WINDOW_PAIRED_TRAIN_REVIEW_20260915.json. Råresultat-SHA:
aab5bacd4a5ba4ba4d885150073e290d43cdfa50d935705add22633a2f1decfa.
Originaler, fire input/targetcacher, batchrapporter, operator og logg ligger i
NATIVE_FQI_TARGET_REFRESH_F11C1DBE/OPERATOR_OBSERVATIONS/
ACTUAL_WINDOW_PAIRED_TRAIN_CPU_20260915_V2 under bundet prebuilt-rot.

Første operatorforsøk92416/exit1 feilet før første modellforward: det krevde
sample.epoch_index==1, mens full TRAIN-epoch1 inneholder interne sampler-chunker
25/26/27 for disse batchene. Første betingelse kortsluttet; ingen avvikende
rekkefølge var målt. Kun operatorens epoch-kontroll ble rettet. Full epoch og
chunk-indekser valideres separat; ordnet sample-digest er uendret. Feilet operator,
logg og COHORT er bevart i første output uten _V2. Ingen produksjonsrettelse.

## Målegrunnlag og neste arbeid

Native TRAIN er deterministisk stokket over hele året, ikke kalenderblokker.
Gjenskapt faktisk epoch_order matcher lagret hash84813100...fbfa65. Begge256-
kontrollene brukte4096entries overalle12måneder (henholdsvis314–367 og313–369
per måned). Ingen shuffle-rettelse er begrunnet.

De64 tidligere proberadene var fire sammenhengende75min entryblokker med eldre
sampler-transitions. Bare4 og5 av entry-IDene forekom i de to kontrollene. Den
små MSE-forverringen i gammel probe var reell på det utvalget, men må ikke omtales
som bevist forverring av hele modellen. Den nye målingen erstatter den gamle som
TRAIN-fit-grunnlag; begge bevares. Ingen av dem er en heldout-kvalitetsgate.
Tidligere target-preflight20.39s og etteranalyse24.63s skal ikke gjentas. Se
FQI_TARGET_REFRESH_CONTROL_RESULT_20260915.json, FQI_TARGET_PREFLIGHT_RESULT_20260915.json
og FQI_TARGET_LEARNING_RESULT_20260915.json i handover_snapshot.

Gjennomgangen av de ni lærer-valgte inngangene er ferdig på lagrede JSON-er.
FLAT ligger rundt0; å sette FLAT eksakt0 ville fortsatt gitt64FLAT. Alle online
LONG/SHORT-verdier er negative. På de ni lærer-valgte sidene er targets+0.481 til
+18.097Bps, etterprediksjon−6.651 til−4.912. Ni rader står allerede for87.69% av
EntryQ-kvadratfeilen; dette begrunner ikke automatisk klassevekting/oversampling.
Side-halvsummen ligger nær lærernivået (−5.708 vs−5.736), mens retningens
halvforskjell har std0.224 mot target5.406. Studenten lærer hovedsakelig det
felles negative nivået på dette utvalget, med lite betinget variasjon. Riktig
LONG-vs-SHORT-rangering på de ni er4/9 etter mot6/9 før. Dette skiller ikke alene
manglende nyttig informasjon fra utilstrekkelig læring eller svake targets.

Eksakt dekomponering fra allerede lagrede input/targettensorer er ferdig,
session36519/exit0,1.207s, CPU4GiB/512MiB. Ingen modell/forward/optimizer eller
nytt markedskorpus. For alle64 er Entry-target eksakt første lukkingsverdi pluss
max gyldig anchor-Q fra lærer91. Alle ni lærer-trader har positiv fysisk første
lukkingsverdi: gjennomsnitt6.431627Bps. Lærerens videre HOLD-bidrag er bare
0.028161Bps i snitt (0–0.045069), mot total6.459788. Fortsettelsesbidraget er
0.43594% av positiv målverdi i disse ni. Dette viser hvor signalet i prøven kommer
fra; det beviser ikke at disse observerte prisbevegelsene kunne forutsies ved
entry eller at læreren er en optimal profittfasit. Se
ACTUAL_WINDOW_ENTRY_GAP_REVIEW_20260915.json og
ACTUAL_WINDOW_ENTRY_ANCHOR_DECOMPOSITION_20260915.json i handover_snapshot.

Neste anbefalte arbeid er en avgrenset lærbarhetskontroll av eksisterende EntryQ
på de samme64 bundne native TRAIN-inputene, fast lærer, uendret MSE, og alle55FLAT-
rader beholdt. Formålet er å skille lokal tilpasningsevne fra svak betinget
informasjon/lærertarget. En eventuell optimizerkontroll må først gis en eksplisitt
avgrenset native plan med eksisterende vakter og bevarte checkpoints; ingen
alternativ treningsvei eller restart av uttømt plan. Ingen produksjonsrettelse,
klassevekter, handelsterskler, ny modell eller hel epoch er besluttet. Evne til å
tilpasse et lite TRAIN-utvalg er heller ingen generaliserings-/profittgate.
Gjenbruk alle fullførte målinger, særlig anchor-dekomponeringen; ikke gjenta dem.
Alle jobber er terminale. Sourcefrys er opphevet kun for ferdig arbeid/commit.
RUNNING_NATIVE_CALIBRATION.json er den operative statusen.

## Historikk: fixed-target-kontrollen87→91

Eneste kodebase er /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Den avsluttede kontrollens frosne kilde er da7ae15d35e4a898656a2c1e6f9519f84b3e9ed5.
Checkpoint91 har global5521, epoch_index1, next_batch_offset1440. Original87
er bevart. Ingen aktiv tung jobb; Windows-task er Disabled. NEXT_RUN_POLICY.json
har training_enabled=false. Den eneste tillatte256-stegskontrollen er brukt opp.
Ingen restart, hel epoch eller automatisk læreroppdatering er tillatt fra den planen.

Native256 steg tok1197.100656sekunder inkludert oppstart og observer, uten fysisk
reboot. Guard PASS, trainer0, observer0. Alle726 Adamtilstander økte256 steg;
EMA25173→25429. Frossen lærer, scheduler og datarekkefølge er bevart. Faktisk
CPU-overføring bevarte alle15 tilstandskomponenter. CPU- og native-kontraktene
avviker bare i run/output/recipe-identitet. Dette er faktisk checkpointoverføring,
ikke en ny sammenhengende-mot-delt-resume-likhetsmåling eller relativ speedup.
De143 målrettede regresjonene gjenbrukes; ingen ny fullsuite.

Paret CPU-evaluering er ferdig på38.254272sekunder, peak2279356KiB, exit0.
Samme64 TRAIN-rader og256 HOLD-celler per side, identiske targets/dropoutfrø;
alle fire checkpoint87-referanser reproduseres eksakt. Ingen autograd, optimizer,
GPU, VAL eller TEST. Endelig operatørskript/logg og en tidligere bindingsfeil
før modelleksekvering er bevart. Se:
- handover_snapshot/FIXED_TARGET_FIT_CONTROL_RESULT_20260915.json
- handover_snapshot/FIXED_TARGET_LEARNING_RESULT_20260915.json
- RUNNING_NATIVE_CALIBRATION.json

## Læringsresultat og neste konkrete justering

HOLD-MSE LONG275.713→274.338, nullbaseline274.046; SHORT268.292→266.439,
nullbaseline266.666. MAE forverres på begge sider. SHORT HOLD212→256 av256;
LONG HOLD89→30. Entry er fortsatt64FLAT mot lærer58FLAT/3LONG/3SHORT.
Dette viser endret tilpasning, ikke dokumentert selektivitet eller lønnsomhet.
Realiserte ettstegsfortegn er ingen feilfri fasit for forventningsverdi. Verken
MAE eller handlingssamsvar alene avgjør om en økonomisk policy er god.

Sekvensinput er allerede til stede. Den konkrete begrensningen i lærersløyfen
er én successor-backup mot frosne verdier, med ny ONLINE-lærer først etter full
epoch og VAL (entry_v10_ctx_train_v3.py, native epoch-overgang). Flere pass mot
samme lærer gjør ikke denne verdipropageringen raskere. Tidligere to-stegsprobe
og gradientkontroll er ferdige og skal ikke gjentas.

Neste anbefalte rettelse er én eksplisitt SHA-bundet FQI-læreroppdatering fra
bevart ONLINE91, frigjort fra full epoch/VAL, før videre avgrenset læringskontroll.
Dette er en kontrollert læringshypotese, ikke godkjenning av lærerens kvalitet.
Bevar gammel lærer og alle øvrige tilstander; kvitter bare tilsiktet target-kopi.
Verifiser smal overgang/resume før eventuell ny native kontroll. Ingen permanent
«hver256»-regel er besluttet. Ingen endring i MSE, økonomi, EMA, Adam, arkitektur,
risiko eller holdetid begrunnes av disse resultatene. Ikke klipp legitime store
utfall for å pynte på fit. Nye hele epocher forblir blokkert.

TRAIN er fortsatt ett år, alle200features/åtte familier/tidsrammer beholdes,
og TEST er forseglet. Nativeprofil TRAIN16/VAL256/8arbeidere/3timersVAL og vakter
beholdes. Historikken nedenfor gjelder avsluttede kontroller, ikke gjeldende
oppstartstillatelse. Nye oppstartsplaner må bindes til gjeldende kilde og policy.

## Historikk: GX1 — stoppet; native kontroll bestått, læring fortsatt utilstrekkelig, 2026-09-15

Eneste kilde er /home/andre2/src/GX1_CURRENT, branch work/gx1-current,
rettelsen er pushet i6ffd03ee13e28fd7ef84e65946ece1b038142166, med utgangspunkt
i stoppet kampanje9cd29a2f57a0880e33c6c0a61b279deb7c15b981.
NEXT_RUN_POLICY.json har training_enabled=false. En smal rettelse i optimizerens
gradientklipping er gjort og målrettet testet. Gammel policy er arkivert uendret.
Originale lagrede modellvekter/checkpoints er bevart. Ny native kontroll er
ferdig:16+16 steg, begge guard PASS og trainer/observer exit0. Checkpoint87 har
global5265/epoch_index1/offset1184. Windows-task er Disabled; ingen native prosess.
CPU-overgangen fra85 består bitnøyaktig på15tilstandskomponenter. Kun16+16
ekstra steg var tillatt (global5249/5265), med eksisterende maskinvarevakter.
Gjenopptakelsen over to fysiske booter består; alle726 Adamtilstander økte16 per
vindu, EMA25141→25157→25173. Ikke ny32-sammenhengende-mot16+16-likhetsmåling.
Se handover_snapshot/EXIT_CLIP_CONTROL_RESULT_20260915.json. Paret CPU-måling av
samme TRAIN-input før85/etter87 er ferdig på410.16s, uten optimizer/VAL/TEST.
Se RUNNING_NATIVE_CALIBRATION.json for plan og runtime. Optimizerkontrollens kilderevisjon
er6ffd03ee; to-stegsproben brukte2c593aa7 med identiske produksjonsbindingsfiler.
Gradientkontrollen er også ferdig på kilde194bb596, uten produksjonsendring.
Ingen aktiv jobb; dokumentasjon kan oppdateres.
TRAIN er 2025-06-01 inklusiv til 2026-06-01 eksklusiv:65 295rader/4081steg per
hel epoch. Hele juni2026 er utviklings-VAL. Ingen ny femårs-epoch. TEST er forseglet.

## Stoppet etter brukerens korrigering

Brukeren avviste videre epoch2-trening før læringsproblemet er målt og en konkret
justering er begrunnet. Ingen automatisk gjenopptakelse er tillatt på grunnlag av
forrige anbefaling. «Ingen kodefeil funnet» er ikke tilstrekkelig læringsbevis.

Windows-taskens fremtidige triggere ble deaktivert13:45:44UTC. Guard726 fikkTERM
13:46:26UTC og stoppet nativePID781. Prosessen er bekreftet borte. Siste checkpoint85
ble SHA-verifisert: epoch_index1,offset1152,global5233,slot0,
stateSHA9f9aaba84a7f3fd031666b9761d606af03b7c534409cf3049021b023805db974.
Dette er operatørstopp, ikke en normal native-vindusfullføring eller guardPASS.
Alle checkpoints/resultater er bevart. Den opprinnelige stoppkvitteringen finnes i
handover_snapshot/RUNNING_BEFORE_CLIP_CONTROL_20260915.json. Gjeldende runtime er
RUNNING_NATIVE_CALIBRATION.json; CPU-diagnostikk og tester er ferdige.

## Målt læringsproblem og minste rettelse

Fire forhåndsvalgte TRAIN-batcher à16 er målt med bevarte epoch1 ONLINE-/targetvekter,
native targets og isolert Exit-backward. 410.50sekunder, exit0, ingen optimizer/CUDA/
TEST; checkpoint-, modell- og policybevaring består. Ikke historisk RNG-gjenspilling.
På256 HOLD-celler per side er LONG255 positive prediksjoner mot120 positive targets;
SHORT20 mot128. HOLD-MSE er274.928/267.108, mot nullbaseline274.571/266.658.
Dette er et fast TRAIN-utvalg, ikke fullårsfit eller generalisering. ONLINE Entry
velger64FLAT, læreren58FLAT/3LONG/3SHORT. Selektivitet er ikke ferdig kalibrert.

Exit-gradienten når head/backbone. I batchen med størst feil bruker tapsvektens
gradient83.58% av kvadrert samlet norm. Felles klipping begrenser derfor også
modellens læring. Rettelsen klipper modell og task_log_variances separat, begge
med eksisterende cap1, etter samlet finite-kontroll. Økonomiske targets, tap,
Adam/EMA-tilstand og risiko er uendret. Ingen crash-targets fjernes eller klippes.
109 målrettede optimizer-/profiler-tester består, inkludert8 nye regresjoner.
Ytterligere134 overgangs-/session-/EMA-/campaign-tester består etter retting
av7 nye fixturetilfeller. Se handover_snapshot/OPTIMIZER_TRANSITION_TESTS_20260915.json.
Se handover_snapshot/EXIT_LEARNING_ADJUSTMENT_20260915.json for råbevis og hasher.
Dette beviser mekanisk rettelse, ikke at Exit-biasen er løst eller modellen profitabel.

Den smale overgangen fra checkpoint85 er verifisert på CPU og native GPU. Den bevarer
modell/target/Adam/EMA/RNG/scheduler/progress og binder ny klippepolicy eksplisitt.
NEXT_RUN_POLICY.json tillater bare16/32 ekstra steg for denne kontrollen; hele
epocher forblir blokkert.32 kontrollsteg og paret TRAIN-måling er ferdige; se nedenfor.
Gamle recipe/policy-bindinger skal ikke omskrives eller brukes til automatisk restart.

## Resultat av kontrollen — ikke grønt lys for hel epoch

Samme64 forhåndsvalgte TRAIN-rader,256 HOLD-celler per side, identisk frozen
teacher/input/dropout. HOLD-MSE LONG275.249→275.713, SHORT267.661→268.292;
begge er dårligere enn nullbaseline274.046/266.666. Fortegnssamsvar øker fra
49.22→54.30% og50.00→53.17%, men HOLD flyttes LONG236→89 og SHORT61→212.
Det er ikke dokumentert robust kalibrering. Før/etter32 steg har ingen gammel-
klipping-kontrollarm og isolerer ikke rettelsens årsakseffekt.

Entry velger64FLAT før og etter; verdilæreren58FLAT/3LONG/3SHORT. Den uavhengige
markedsprognosens MAE faller på alle fire eksisterende horisonter, men dette er64
TRAIN-rader før kostnader. På60min er68.75% retningssamsvar svakere enn utvalgets
alltid-opp-baseline76.56%. Ingen generalisering eller profitabilitet er bevist.
Se handover_snapshot/EXIT_CLIP_PAIRED_TRAIN_RESULT_20260915.json for råmåling.

Neste prioritet er det målte svake videreverdisignalet: de to native batchene
viser frozen videreverdi omtrent0.044Bps mot umiddelbar lukkeverdi omtrent−5.8Bps.
Dette forklarer kostnadsdominert Entry-lærer på disse batchene; det beviser ikke
at mulighetene er profitable eller at hyppigere læreroppdatering løser problemet.
Kildegjennomgang bekrefter successor=state_index+1 i
unified_exit_random_access_sampler_v1.py:233, én-stegs reward+frossen bootstrap i
unified_exit_fitted_q_v1.py:309 og Entry-koblingen i
unified_exit_random_access_training_v1.py:564. Læreren kopieres fra ONLINE først
etter komplett epoch/VAL i entry_v10_ctx_train_v3.py:14052–14058.32-stegskontrollen
endret derfor ingen frozen targets; forverret MSE beviser ikke en bedre lærer.

To-stegsundersøkelsen er nå FERDIG, CPU401.32s/peak8 462 888KiB/exit0. Den brukte
samme64 TRAIN-rader/256 HOLD-celler per side og samme frozen teacher. Ingen
optimizer, ONLINE-forward, GPU, VAL, TEST eller produksjonsendring. Opprinnelige
inputs og alle fire originale target-SHA-er ble reprodusert eksakt; kilde,
checkpoint, RNG, sampleplan, policy og modellvekter er bevart.

Resultat: LONG254 unik HOLD og2 unik EXIT ved første successor; SHORT20 HOLD
og236 EXIT. Ingen ties, reelle terminaler eller manglende neste tilstand i dette
utvalget. EXIT-grenen gir eksakt samme mål. Derfor kan flere steg med samme
frosne policy ikke tilføre videre reward i236/256SHORT-tilfellene. Ingen n-stegs-
treningsendring er begrunnet eller innført. Backup-lengde er ingen holdegrense.

Med identiske bevarte ONLINE87-prediksjoner mot et annet mål går LONG-MSE
275.713→191.065, men målspredningen faller16.549→13.828Bps og nullbaseline
274.046→191.353; MAE øker4.086→5.448. Dette er endrede targets, ikke læringsgevinst.
SHORT-MSE268.292→268.621. Ingen optimal n, generalisering eller profitt er bevist.
Se handover_snapshot/TWO_STEP_TARGET_PROBE_20260915.json. Råbatchene og
operatørskriptet ligger i kjøringens OPERATOR_OBSERVATIONS/TWO_STEP_TARGET_PROBE_CPU_20260915.

Nær-Entry-supervisjon mangler ikke:34 av256 sampled states erstate0,60 erstate1–15
og162 senere. State0 har egen bucket. Ingen ombygging av no-loss-ankeret nå.
Sekvensinput er også til stede: Entry96 historiske basebars og egne HTF-sekvenser;
Exit480 historiske M1-bars og kausale MTF-/trade-path-inputs. Entry bruker
transformerencodere; Exit sin nåværende sekvensrute bruker egne GRU-er og
familieattention. Dette er eksisterende arkitektur, ikke en ny endring. Sekvenshistorikk er
ikke det samme som hvilken framtidig økonomi treningsmålet overfører bakover.

Gradientkontrollen er nå FERDIG: V3, CPU 436.926 s, peak 12 697 500 KiB,
exit0 under eksisterende producer 20G/512M. Samme fire TRAIN-batcher, checkpoint87,
faste SHA-bundne targets og native loss-vekting. Ingen optimizer, GPU, VAL eller
TEST. Alle fire originale no_grad-ankre reproduseres eksakt; native CPU-gradientbane
har maksimalt 9.5367e-7 Bps prognoseavvik, ingen endrede Entry-/Exit-handlinger,
eksakt slutt-RNG og identiske rå forecast-/Exit-tap. Dette er ikke GPU-paritet.
Kilde, policy, checkpoints, modell, targets og sampleplan er bevart.

Faktisk Exit-/forecast-overlapp finnes på 138 parameterobjekter. Cosinus er
−0.01193, −0.02251, +0.04800 og +0.01727. Exit-norm på denne støtten er
0.001308/0.000700/0.054388/0.001666 mot forecast 0.823/0.340/0.385/2.716.
Samlet dot(forecast, non-Exit + komplett Exit) er positiv i alle fire batcher:
19.563314, 3.042481, 9.281150, 110.414097. Alle 324 Entry-private encoderparametere
og fire beskyttede rutingsparametere er uten Exit-gradient. Exit-tokenets
reinjeksjon når bare fire parametere i tokenprojeksjonen. Lokale gruppekonflikter
finnes, men disse målingene støtter ikke at Exit samlet ødelegger forecast-læringen.
Rå gradientgeometri er ikke bevis om faktisk Adam-steg, generalisering eller profitt.

Behold arkitektur, gradientgrenser, tapsvekter og eksisterende clipping. Denne
gradientdiagnosegrenen er avsluttet og skal ikke gjentas. Se
handover_snapshot/SHARED_TASK_GRADIENT_RESULT_20260915.json og
handover_snapshot/SHARED_TASK_GRADIENT_REVIEW_20260915.json. Råbatcher, eksakte
inputcacher, skript og logg er bevart under OPERATOR_OBSERVATIONS/
SHARED_TASK_GRADIENT_PROBE_CPU_20260915_V3. Inputcachene gjør nye relevante
avgrensede kontroller mulige uten å laste hele korpus på nytt.

V1 og V2 stoppet før gradientmåling ved krav om eksakt EntryQ-replay. V3
reproduserte eksplisitt originaloperatørens frosne parameterflagg i no_grad-ankeret
og gjenopprettet native flagg før gradientuttak. Alle fire ankrene bestod da.
Lavnivåårsaken til tidligere avvik er ikke isolert; ingen produksjonsfeil eller
korrigert GPU-paritet hevdes. Feillogger/skript er bevart, ikke overskrevet.

Neste arbeid er én avgrenset native læringskontroll rettet mot Exit-critic og
verdilæreren. Definer kontrollen fra checkpoint87 med eksisterende TRAIN-evidens:
vis først bedre tilpasning mot faste targets før en oppdatert lærer tas i bruk.
Dagens lærer oppdateres først etter hel epoch og full VAL; dette er en mulig
flaskehals for videreføring av verdi gjennom etterfølgere, ikke en bevist eneste
rotårsak. Hyppigere læreroppdatering må måles kontrollert og skal ikke innføres
blindt. Ingen ny full epoch, målfrekvensendring eller gjentakelse av ferdig
32-stegskontroll følger automatisk av denne rapporten. Bruk eksisterende native
campaign/profil/vakter; ingen alternativ treningsløype, brede regel-/modelltester
eller vilkårlige risiko-/holdetidsgrenser. Windows-task er Disabled. TEST er bevart.

Metodebakgrunn: policy-evaluering med fler-stegsretur og off-policy-korreksjon,
Munos mfl., https://arxiv.org/html/1606.02647v2 (seksjon1–2). Vår to-stegsprobe
fulgte en frossen greedy-policy og valgte aldri den beste realiserte exit-tiden
i etterkant; ingen generell konvergensgaranti for GX1 ble utledet fra artikkelen.

## Første ettårs-resultat og avgrenset ONLINE/EMA-sammenligning

Hele juniVAL er ferdig og klart negativt. Entry valgte4207LONG/1301SHORT/0FLAT.
Én-posisjonsreplay beholdt første handel til månedsslutt:0modell-exits,1åpen,
5507hoppet over. Kostnadsjustert cash pluss åpen verdi er−1235.8772Bps på fast
nominelt beløp, ikke sammensatt kontoavkastning. Uavhengige entrymuligheter har
snitt−495.3189Bps, ikke porteføljeresultat. Lukkede vinnere er ikke samlet profitt.
Kontrafaktisk lukkes alle5508SHORT vedstate0; LONG785EXIT og4723HOLD tilsplit-end.
Retningsprognosen treffer omtrent48–51% på5–60minutter i denne måneden.
Se handover_snapshot/EPOCH1_FULL_VAL_20260915.json. FullVAL71 315 567tilstandsvisninger/
291294forwards/42449.66beregningssekunder; samlet relativ speedup er ikke målt.

Bevarteepoch1-vekter ble sammenlignet på samme CPU-input: ONLINEvelgerFLAT påalle8
fasteEntries; EMA7LONG/1SHORT. Exit har samme sidehandlinger ibegge på10tilstander.
EMA beholder81.19% parametervekting fra start etter4081steg, men forklarer ikke
Exit-skjevheten alene. Ingen EMA-policy er endret. Åtte rader er ikke full ONLINE-VAL.

CPU-sanity forblir REVIEW_EMA_REFERENCE_MISMATCH: CPUbatch8 mot GPUEntrybatch16 har
max0.00012672Bps Q-avvik. Alle8EMA-handlinger matcher; minstemarginEMA0.20565Bps/
ONLINE4.48361Bps. Ingen konkret input-/normaliseringsfeil funnet. Det kvalitative
sammeCPU-funnet er tydelig, men numerisk paritetPASS og lønnsomhet hevdes ikke.
Toleransen er uendret; ingen gjentatt kjøring er nødvendig for dette funnet.
Se handover_snapshot/EPOCH1_ONLINE_EMA_COMPARE_20260915.json.

## Bevarte porter og arbeidsregler

Native porter på128c55f2 er bestått: faktisk overgang fra original315, GPU256-
paritet, målt absolutt fart og eksakt32mot16+16 resume på14tilstandskomponenter.
Dette er historiske bevis fra før optimizerrettelsen. Gjenbruk uendrede inferensbevis;
ny optimizer-/checkpointovergang må verifiseres før videre trening.
Gamle femårsresultater/original315og19 908steg er bevart; gammel juni analyseres
bare med første gamle epochs uforanderlige EMA. Gamle kildekopier er avhengigheter.

Ingen fast holde-/tapsgrense. Bevar alle200features,familier,tidsrammer,kostnader,
successor-semantikk og checkpoints. Én tung jobb samtidig; underagenter er autorisert
for avgrenset arbeid, men ingen sideoppgaver under venting. Stabil drift sjekkes
omtrent hver time. Stående autorisasjon gjelder. Ingen TEST/live/papir/spending.

Oppdater handover ved vesentlig endring og commit/push ferdig rettelse med bevis.
Resultatbevis og alle gamle kjøringer bevares. Eldre driftsdetaljer er bevart i
handover_snapshot/CURRENT_HANDOVER_BEFORE_EPOCH2_RESUME_20260915.md.
