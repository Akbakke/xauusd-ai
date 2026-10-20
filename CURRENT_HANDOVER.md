# CURRENT HANDOVER — XAUUSD M5 Entry, kompakt førsteprofil

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les GX1_RULES.md og kjør `bash scripts/gx1_handover.sh --check`.
NEXT_RUN_POLICY.json eier arbeidsstatus og eksakte artefaktbindinger.
Én integrator, avgrensede agenter, én capped tung jobb og fryst kilde/policy.
Originalkvitteringer bevares; konsumerte engangsplaner relanseres aldri.

## Mål og beholdt omfang

M5 Entry skal lære LONG, SHORT og FLAT fra observerbar markedstilstand.
Retning/netto endepunktsretur, signert MFE og konservativ MAE før første MFE
måles separat fra ett minutt til åtte timer; dager er støtte. Ingen Exit-
vurdering eller Exit-trening. Horisonter er ingen maksimal holdetid.
Behold lukket M5/M15/H1/H4/D1 og de rikeste26månedene med GC MBP-1.
Kvalitet og dekning skal måles for både valgte og avståtte entries.

Første profil er `compact_mbp_smc_ema_atr_clock_macro_entry_v1`: MBP-1,
SMC/struktur, EMA/ATR og nødvendig klokke-/makrokontekst. Utvalgte candle-/
chart-pattern- og overlappende felt er parkert. Hele biblioteket og fullmodellens
åtte familier med niende D1/makrokontekst beholdes. Ingen felt er erklært støy
uten måling. Separate SMC/MBP-modeller, ensemble og M30 krever senere samme
kronologiske OOF-evidens for merverdi.

TRAIN er [2024-05-01,2026-01-01), VAL [2026-01-01,2026-04-01), senere
utviklingskontroll [2026-04-01,2026-07-01). Alle26måneder er gjenbrukt utvikling;
TEST fra juli er forseglet. Den bestilte ene nativeepoken er FULLFØRT i108.
Ingen gjenværende treningsadgang: training_enabled=false/fullEpoch=false/fullVAL=false.
Ingen broker/live/paper/ordre/spending/promotion. De to godkjente practice-
leseoppslagene er konsumert. Ingen Exit-vurdering eller Exit-trening er utført.

## Aktuelt — hel epoke ferdig; Entry-kvalitet er ikke kvalifisert

108root: `/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_SINGLE_FULL_NATIVE_EPOCH_20261020_108`.
Faktisk én ordnet TRAIN-epoke:118249unike rader,7391oppdateringer,batch16,siste9.
Canonical kjøring20.10kl.13:16:01–14:57:31UTC,guardPASS/trainer0/observer0.
Kun ett av tre forhåndsbundne vinduer ble brukt; de to andre gir ingen ny adgang.
Generic RESUMABLE/epoch0/complete=false betyr ingen ekstraepoke eller fortsettelse.
Windows-task er Disabled/LastResult0, ingen bundne prosjektjobber eller ACTIVE.

ACTUAL_SINGLE_EPOCH_TERMINAL_REVIEW.json
SHA87cf966c9cbbecd3542cd16e9d374a45fb10b346474f27c9b5caf7bb7693e543.
PAIRED_LEARNING_REVIEW.json
SHA3b360860d7d80ba8de0d8f511c912afc3726570ebba206082a04f79323db4c9d.
POST_NATIVE_SOURCE_PRESERVATION.json
SHA46eb57e1ed81968a532aa9440089229ef50aacfb705ce0601fe9b355eca339ad.
WINDOWS_TASK_DISABLED.json
SHA335627df0caa981dafe1a76d1e70b0d6cb35c4b0bda8c621f13d17f69fd80fda.

Fast ONLINE-sluttmåling ble sammenlignet med105førmåling på nøyaktig samme
256TRAIN+256CONTROL-rader og seks uendrede target-/maskeflater. Ingen utvalg
eller tilpasning i reviewet. Alle seks samlede TRAIN-komponentfeil falt.
På CONTROL falt samlede MFE-/første-MFE-MAE-prognosefeil, mens endepunktsretur
og støtteforecast ble dårligere. Ettminuttsfeilene økte på begge probegrupper.
Alle512sluttprober velger diagnostisk1minutt som beste predikerte horisont
for begge sider. Kandidatfordeling LONG/SHORT/FLAT er TRAIN0/8/248 og
CONTROL0/108/148. Dette er kandidatdiagnostikk, ikke handler eller akseptert
native beslutningsautoritet. Delvis komponentlæring er målt; hovedmålets
Entry-kvalitet, generalisering og økonomisk edge er fortsatt ikke kvalifisert.

## Ferdig — gjenbruk data og beståtte kontroller

- 66GC-råfiler med892100712records og9173893prisnivårader er bevart. Genuine
T-poster dedupliseres ikke;17PTP-minutter/18bok-eventbøtter er sensurert.
MBP-1 viser BBO/ordercount, ingen skjult dybde eller ordreidentitet.
- 1205GCverdier+1205separate masker er publisert. Fysisk flate er
signal2725/ctx-cont2542/ctx-cat1 og253felt per TF.051kilderekonstruksjon og
053input-liveness bestod på118249TRAIN-/17284VAL-rader. Ingen ny rebuild.
- 055wholeTRAIN Entry-only-normalisering er ferdig, SHA
c02ca97967a314f7dbd6c8bebc341c9782acd9e999e67e4b730db7e99e318c62.
Hele transformen og domene-/finite-/alias-/maskevakter brukes før gather.
- 057binder utfallsblindt valgte256TRAIN-/256CONTROL-prober og ordnet full
TRAIN-populasjon.059kostvilkår er prospektive, ikke historisk kostsannhet.
- Kompaktprofil beholder alle1205GCverdier+masker og61D1/makrofelt.
Aktiv flate2622signal/2525ctx/139MTF,7lokale encodere og25familie×TF-ruter;
candle-ruten beholder74genuine MBP-kanaler. Lokal96bar og MTF16/64/96/96/252.
- 104fresh→105initial32Entryforwards/0updates→106strictsavedstate/RNGaudit
bestod på aktuell closure.107bestod fire private Entry-oppdateringer,
fire paritetsforwards og åtte eksaktlike resume-state-trær; outputavvik0.
Private107vekter ble aldri brukt i108.103rettelsen bestod148fokuserte caser.
- 108endret ONLINE-state til542cf139…; opprinnelig anker9c4cf514… ble bevart.
182kilderoller/181filer og11inputstatidentiteter er kontrollert bevart.
Funksjonsclosure dc1c6b627a2d42a9fea18fcd5dd6bdcfca493a840f32bbf01e1a765a919b785d.

## Målte utfordringer og neste steg

De konkrete attention-/Entry-oppstartsfeilene er rettet og kvalifisert.
109 er nå COMPLETE: avgrenset CPU-audit på 1,93 sekunder, to lagrede checkpoints,
fire faktiske ONLINE-taskskalarer hver, ingen modellbygging/forwards/updates,
GPU, refit eller rådatasettlesing. TRAIN- og CONTROL-probene har256rader hver.

109root: `/home/andre2/GX1_RUNS/NATIVE_M5_TASK_SCALE_AUDIT_20261020_109`.
TASK_SCALE_AUDIT.json SHA97cda80ee2c2cea53015a50f60a5d66bf6af524ae163b7ba8a22c38d34ec2b5d.
PAIRED_SHORT_HORIZON_PANEL.json SHAddcc5c186bb1760f5f12f97e894113392f18a2083c616c55c2cabe2a866bbdf2.
TASK_SCALE_ARITHMETIC_REVIEW.json SHA30fa1051f18294737f8e36f33cdafb1bb313e06bae17780f84abcc23d89c25dc.

Alle fire lærte presisjoner ligger mellom0,47804 og0,49442. Dagstøtte utgjør
82,0985%TRAIN/93,3483%CONTROL av fire taskers vektede, cellepoolede residual
ved sluttmålingen. Dette er målt endpoint-lossverdi, ikke parametergradient
eller årsak til h1-forverringen. Forecastens felles presisjon endrer ikke
forholdet mellom M5- og dagskoordinatene innad i forecast-tasken.

Det faste panelet1/5/15/60/240/480minutter gjenbruker lagrede mål/masker og
feilsummer; forecast har sine egne10koordinater, ingen falsk480akse.
På CONTROL har de108diagnostiske SHORT-valgene gjennomsnittlig h1-nettotarget
−6,58996bps; FLAT-gruppens148hypotetiske SHORT har−7,06159bps. Disse er
lagrede, deklarerte mål og ingen utførte handler eller bekreftet økonomisk edge.
Kandidaten bruker max_h(terminal(h)−preMAE(h)) for samme h per side; arkitekturen
tvinger ikke h1, og ingen etterfølgende horisontveto eller filter innføres.

110 er COMPLETE for loss-only TRAIN-RMS-kilde og avgrensede kildekontroller.
DESIGN.json under `/home/andre2/GX1_RUNS/NATIVE_M5_TRAIN_LOSS_RMS_SOURCE_20261020_110`,
SHAa4042a68b4b3931a374a7791e2fd87bef9bb7fb2a4184d271b0d90026bd87840.
RMS=sqrt(sum(target_float32² i float64)/observed_count) tilpasses separat for
4810 lærbare koordinater:480×2 terminal-L/S,fire480×2pathflater og10forecast.
Kun feilene skaleres før eksisterende pooling; rawbps-targets/prognoser/masker
og rå feilrapportering beholdes. FLAT0 bypasser divisjon med gammel denominator.
Count0/RMS0/ikke-finit skala feiler; ingen sentrering, epsilon eller unitfallback.

Ny genuin statistikkeier er entry_dynamic_m5_loss_scale_v1.py; eksisterende
mål-eier deler en target-only kildefabrikk. Recipe→dataset→targetbatch→unitloss
og session/resume/initial-binding er implementert og kildekvalifisert. StatistikkCLI skal lese
bare26targetkolonner og originalt M1-quoteprefix, ingen features/Dataset/modell.
Ferdige inputs/targets/055inputnormalisering gjenbrukes. Fire capped kildejobber
bevarte samme produksjonskilde; bare konkrete testfixtureavvik ble rettet.
Beståtte caser gjenbrukes, siste berørte runde hadde300PASS/3eksplisitte
nonTRAIN-derived-SKIP. Alle11berørte moduler er dekket; ingen ny fullsuite.
SOURCE_QUALIFICATION_REVIEW.json og TERMINAL.json i110binder bevisene.
Syntetiske tester er ingen faktisk TRAIN-fit. Ny111fit er nå faktisk COMPLETE:

111root: `/home/andre2/GX1_RUNS/NATIVE_M5_TRAIN_LOSS_RMS_FIT_20261020_111`.
Én capped CPUaudit4G/512M gikk gjennom118249TRAIN-rader én gang i924chunks.
Oppstart83,049s,fit129,663s,total212,793s; peakRSS3078268KiB. Alle4810
koordinater har positive observedcounts og RMS; råsqrtaritmetikken ble
uavhengig kontrollert fra publiserte summer/counts uten targetrebuild.
LOSS_SCALES.json SHA93fa60c423a4adebd888f7a905b31adc61a3094d8e7a04169f4049db540fc88d.
PUBLISHED_SCALE_REVIEW.json SHA8dbaa6c22a04fc30b5327c5e053071d1412e078500ca28e23c098846b827169d.
172kildebindinger og25metadata-inputidentiteter er bevart. Ingen ny featurebuild,
inputnormalisering,model/forward/optimizer,GPU,VAL-fit ellerTEST.111er konsumert;
artefaktet gjenbrukes og refittes ikke påVAL. Dette er ikke bedre prediksjon.
Siste terminal er112, kontrollert avbrutt for målt metadata-I/O.
112root: `/home/andre2/GX1_RUNS/NATIVE_M5_TRAIN_RMS_FRESH_ENTRY_20261020_112`.
En korrekt bundet CPUfresh-jobb startet17:06:48UTC; integratoren stoppet eksakt
worker medSIGTERM etter gjentatte inputkontroller før publisert freshstate.
Controller avsluttet17:23:20UTC etter992,515s,worker143/FAILED_NO_ADMISSION.
197kilder ogpolicy er bevart; alle bundne prosesser er borte. Ingen freshstate,
RESULT eller workerterminal ble publisert. Modell-/TEST-tellere er ukjente
etter avbruddet; ingen nuller eller ferdigstilling konstrueres.
CONTROLLED_STOP_REVIEW.json SHA612fa1ee7c85e8a337f2aa879b862d50187c4d77e9d644e8b62c337242aaaae0
binder kontrollert stopp og opprinnelig controllerkvittering.
TERMINAL.json SHA6cc9deb27ab09223d06fca0fd6de49da971bd806b63f18fdae4d913b6fcb6180 normaliserer
controllerens faktiske worker143 til handoverformat; statusFAILED_NO_ADMISSION
og ukjent TEST-observasjon beholdes. Dette er ingen workercompletion. Prosessens
read_bytes var554317824832 og rchar564945348985 ved stoprequest; dette er
Linux-prosesstellere, ikke hosttotal eller bevist mengde duplikatlesing.

Kilderevisjonen fant at publicfresh recipe/prefixadmission mangler samlet
finite metadata-scope, og _bound_artifact bruker rå checkpoint-hasher i stedet
for eksisterende common-hasheier. Senere metadata etter state-roundtrip mangler
eget scope. Minste neste rettelse er common-inputhash pluss separate scopes
før modellbygging og etter strictstate/RNG. Ingen memo gjennom modell/state;
råde PT-hasher bevares.112er konsumert og relanseres aldri.111RMS,11inputs,
profil/targets og055normalisering gjenbrukes etter ny kildekvalifisering og
ny unik fresh-/initialbinding. Den tidligere113initialstien ble aldri laget.
Ingen ny epoke åpnes;108har allerede oppfylt den bestilte ene epoken.

113 er COMPLETE for den avgrensede metadata-I/O-kilderettelsen.
Root `/home/andre2/GX1_RUNS/NATIVE_M5_METADATA_IO_SCOPE_REPAIR_20261020_113`.
Én capped kildejobb4G/512M tok36,007s:102PASS,0fail/error/skip. Alle8nye
caser kjørte.201kilder ogpolicy er bevart; uavhengig XML-/bindingkontroll bestod.
SOURCE_REVIEW.json SHA6fad1a14e4658276ae2cf6e30c1a461009cc031405e0474a867f2044f9038e47
binder faktiske resultater. Hele produksjons-AST er identisk med forrige
kilde etter normalisering av kun hasheralias/-kall og to with-innpakninger.
Rawcheckpoint-hasheieren er uendret. Små faktiske filstrømmer og syntetisk
TinyModel/state beviser gjenbruk, feilavvisning og grenser; faktisk I/O-gevinst
er fortsatt umålt. Påkrevde commit-kontrakttester bestod også før113.

Neste steg er NY114CPUfresh med113kvalifisert kilde,111TRAIN-RMS og faktisk
prosess-I/O-/tidsmåling. Ingen114plan eller launch er ennå bundet.11inputs,
055normalisering, profil, masker og targets gjenbrukes.112/113er konsumert;
ingen relaunch. Nye initial-/paritets-/resumekontroller krever senere egne
eksakte bindinger. Ingen ny epoke eller markedsforwardadgang er åpnet.

VAL-targetstd er aldri fitgrunnlag, og CONTROL-resultater velger ingen skala.
Ingen ny modell, håndsatt relativ vekt eller ekstraepoke.109og108er konsumert.
Etter endret sourceclosure er105/106/107 historiske funksjonsbevis; ny aktuell
CPUfresh/initial/paritet/resume må kvalifiseres før eventuell ny trening.
111targetfit er fullført; nye modellforwards og trening er fortsatt stengt
inntil egne eksakte bindinger/kvalifiseringer. Ingen gamle kampanjer eller
initialmålinger relanseres.

Alle1205GCverdier+masker har kildekart for formel/enhet/domene/warmup;
23tekniske formler×5TF har115feltposter. Dette er ikke full uavhengig numerisk
kontroll eller dokumentert nytte av hvert felt. Konkrete gjenværende kontroller
står i docs/FEATURE_AUDIT.md. Native LONG/SHORT/FLAT-kvalitet, valgte/avståtte
entries, horisontkalibrering og beslutningsforklaringer må fortsatt kvalifiseres.
Makroens tidligere oppgitte13felt med manglende originalvintage er ingen full
point-in-time-kvalifisering. TEST er fortsatt forseglet og urørt.

## Maskin, effektivitet og opprydding

108bekrefter fysisk Windowsboot516/20.10kl.13:09:26.500UTC,
REBOOT_CONFIRMED.json SHAc63ea6791d64a1cae1689c63944c09dd405ec29d69653c3dcf2c1fca9a4bdc51.
18.10-omstartbestillingen er også allerede fullført. Ingen ny omstart nå.
Signerte108maksima var57°Ccore/66°Cminne/133.84W/1802MiB; vaktene bestod.

Hele108tok6090.176sinkludert oppstart/avslutning; sessionstart til siste
checkpoint3910.443s, inklusive loader/checkpointarbeid. Ingen ren GPU-benchmark.
Ferdig MTF-NPY, targets og normalisering ble gjenbrukt. Ingen tett sekvens-
memmap ble bygget. Sparse prosessmålinger viser små ekstra fysiske lesinger
under stabil trening; rchar er logiske lesinger, ikke fysisk hosttotal.
Kontrollert fartsgevinst er ikke målt. Full fysisk flate og historisk warmup
leses fortsatt ved oppstart. Signal-only retention av unødvendig context/cat-
backing er en konkret, foreløpig uimplementert I/O-kandidat.

DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Git eier historikken.
Metode/bruksanvisning: docs/NATIVE_LEARNING.md. Status står bare her og i
NEXT_RUN_POLICY; regler i GX1_RULES.md, automatisk lesing i gx1_handover.sh.
