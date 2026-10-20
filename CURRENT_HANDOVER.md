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
TEST fra juli er forseglet. Én full nativeepoke er bestilt og gjenstår etter
faktiske profilbundne porter. training_enabled=false; fullEpoch/fullVAL=false.
Ingen broker/live/paper/ordre/spending/promotion. De nøyaktig to godkjente
practice-leseoppslagene er konsumert; ingen gjenværende brokeradgang.

## Ferdig — gjenbruk data og beståtte kontroller

- 66 GC-råfiler med892100712 records og9173893 prisnivårader er bevart.
  Identiske genuine T-poster dedupliseres ikke.17 PTP-minutter/18 bok-event-
  bøtter sensureres. MBP-1 viser BBO/ordercount, ikke skjult dybde/ordreidentitet.
- 1205 GC-verdier med1205 separate masker er publisert. Full fysisk flate er
  signal2725/ctx-cont2542/ctx-cat1 og253 felt per TF.051-kilderekonstruksjon
  og053-input-liveness bestod på118249 TRAIN-/17284 VAL-rader. Ingen ny rebuild.
- 055 whole-TRAIN Entry-only-normalisering bestod; ingen Exit/VAL/TEST-fit.
  SHA c02ca97967a314f7dbd6c8bebc341c9782acd9e999e67e4b730db7e99e318c62.
  Hele transformen og domene-/finite-/alias-/maskevakter beholdes før gather.
- 057 binder native design, mål, koordinater og frosne256TRAIN-/256CONTROL-
  rader valgt uten utfallsutvalg.118249 TRAIN-rader gir7391batcher, siste9.
  059 binder prospektive kostvilkår, ikke historisk kostsannhet.
- Kompaktprofilen beholder alle1205GCverdier+masker og61D1/makrofelt.
  Aktiv flate2622signal/2525ctx/139MTF,7lokale encodere og25familie×TF-ruter;
  candle-ruten beholder74genuine MBP-kanaler. Lokal96bar-sekvens og MTF-
  vinduer16/64/96/96/252 beholdes. Fysisk I/O er fortsatt full flate.
- Ferdige datakilder rekonstruerer sekvenser uten ny tett sekvens-memmap:
  faktisk102logg rapporterte123.74GB TRAIN og18.09GB VAL unngått ekspansjon.
  Dette er beregnet unngått materialisering, ikke målt disk-I/O eller GPU-fart.
- 098s kilde-/syntetiske kvalifisering bestod219caser/3eksakte legacy-SKIP.
  Faktisk099CPUfresh tok121.070s,0forwards/updates/refits, closure8517…;
  faktisk100initialmålte256TRAIN+256CONTROL,32Entry-forwards/0updates,
  1869.792s.101strictsavedstate/RNG bestod på62.220s,0forwards/updates.
  Disse originale bevisene er bevart;103 endrer trainerclosure og krever ny
  faktisk kildebundet CPUfresh og videre eksisterende native porter.

## Aktuell grense —103 bestått;104 CPUfresh ferdig;105 initial0 skal bindes

102s separate nativekvalifisering stoppet20.10 kl.07:34:11UTC med guard
child_status1 og eksakt Windows-task LastTaskResult1. Ingen bundne Linux-
prosesser ved07:45:46; exact102task ble deaktivert07:48:19. Ingen relansering.
Den kanoniske invokasjonskvitteringen mangler. TERMINAL_RECONCILIATION.json
skiller faktisk guard/taskstatus fra en ikke innhentet direct-wait-status.
SHA df368a32819ae339fb12129de0affe938d3aab33388a01d6009296353ac79350.

Tracebacken når første UNINTERRUPTED-segment etter begge paritetsløkkene.
Fire parityforwards er dermed dokumentert fra kilde/kontrollflyt; endelige
numeriske paritetsdiagnoser og full RESULT er ikke publisert. Referansecheckpoint1
har global_steps0/next_batch_offset0. Før første treningsbatch kalte løkken
Exit-lifecycle-setteren som dynamisk M5 Entry uttrykkelig ikke har bundet:
UNIFIED_EXIT_LIFECYCLE_V2_NOT_BOUND.0treningsbatch-forwards/0optimizerupdates;
varig resume-forløp ble ikke nådd. Ingen lærings- eller selektivitetskonklusjon.

182kilders SHA/stat,40statiske artefakter og11inputidentiteter er bevart mot
prebootsegl; ingen ny rådatastreamhash i etterkontrollen.
POST_FAILURE_SOURCE_PRESERVATION SHA23de07564c5f3b748edf6cbd28ad29e37771813b5d9e702f3226f4d78402397d.
Originalclaim, privatstep0checkpoint og alle kvitteringer bevares. Private
kvalifiseringsvekter får aldri brukes til hovedepoken.

103 retter bare den målte epochkoblingen og konkret metadatadobbeltarbeid:
verifisert dynamisk M5-prefixfase hopper over Exit-epoch-setteren; øvrige faser
beholder ekte binding/fail-closed feil. Ett kort eksisterende hashscope omslutter
policybygger+require; begge semantiske kontroller beholdes og scopet lukkes før
artifact-/Session-/statearbeid. Ingen annen top-level trainer-AST er endret.
103s fokuserte capped kvalifisering bestod148unike caser/0skip/0feil på83.270s,
inkludert alle5nye caser for dynamisk fresh/resume, strict legacy og finite
metadata-scope. Direct controller exit0; sourcebefore=after. SOURCE_REVIEW.json
SHA89d506544e9193050b474cc042d66d48d02d4dadd844a1d84c898c6f75929521.
Ny canonical closure dc1c6b627a2d42a9fea18fcd5dd6bdcfca493a840f32bbf01e1a765a919b785d
har182roller/181filer/7154326B; baretrainer er endret mot102s8517-recipe.
Fersk CPUstate og videre native porter gjenstår. Rettelsens faktiske I/O-/
tidsgevinst og native optimizer/resume er fortsatt ikke bevist.

## Individuell indikatorrevisjon

Alle1205GCverdier og1205separate masker har nå kildekart for formel, enhet,
domene og warmup: tidligere350 pluss855disjunkte felt.23tekniske formler×5TF
har115feltposter. Aggregate under NATIVE_M5_READONLY_FORMULA_UNIT_WARMUP_20261020_002:
GC_ALL1205_SOURCE_UNIT_WARMUP_COVERAGE.json,
SHA2dd5c8c4b1dbe16b959d5ae83e145ef123a28db8a9e808e9575a8c11c21026a5.
Dette fullfører kildekartet, ikke uavhengige numeriske bevis for alle felt,
aktuell rådatareferanse, individuelt modellbidrag, læring eller økonomisk edge.
Se docs/FEATURE_AUDIT.md for konkrete resterende bevisgrenser.

## Maskin, effektivitet, neste steg og opprydding

Seneste faktiske separate profilboot er WindowsBoot513/20.10 kl.06:55:31.500UTC;
102/REBOOT_CONFIRMED.json SHA9d1d33b60cdfe640512f33595f648592973aa90e3e40717eaad9075637b1b5ab.
Brukerens18.10-omstart er også allerede utført. Ingen av disse bestillingene
skal gjentas. Ny nativecampaign følger egne gjeldende ferske boot-/hostporter.
Aktiv102effektgrense var300W og drawstop310W, core85°C/memory80°C/VRAM12288MiB;
ACTUAL_ACTIVE_HARDWARE_ENVELOPE.json korrigerer eldre generisk160W-prosa.
Kanonisk hardware-eier og signedguard, ikke dokumenttekst, eier grensene.

Prioriter effektiv datalesing: gjenbruk ferdige inputs/targets/normalisering;
skill engangsvalidering, modelloppsett, batch-I/O, GPU-steg og checkpoint.
En enkelt stack viste sourceverifisering ved coordinatoroppstart, ikke en
batch-/480løkke. Intermediate rchar er logiske lesebytes; read_bytes er separat
lagringsregnskap. Ingen TB-per-batch-, farts- eller ETA-påstand er bevist.
103 samler det konkrete doble policyarbeidet i samme nødvendige kildebølge.

104 er faktisk COMPLETE/exit0/sourceunchanged20.10 kl.08:20:50UTC på120.688s.
Først bestod hele metadata-prefix252–306 på0.208s, utenclaim/Torch/modellimport.
Deretter ble genuinCPUfresh publisert:0forwards/updates/refits/Exit, ingen gamle
vekter.182kilder SHA/stat,11inputidentiteter og71,872,094Binitialblobc62cc1ad…
er fysisk kontrollert; constructor ONLINE/anker/EMA9c4cf514… er lik099.
INIT f5a78482f6c85d85301d4e2f10efcb2a7f6e1ad740a1582a7a15f6380852586f,
MEAS8f3351bda4306808b30afa312be64ceac5664041c2f7b4cf1fe09083c67b5283.
WorkerIO49.232GBrchar/44.555GBread_bytes er ulike roller, ikke hosttotal eller
målt nativegevinst.104 er konsumert; originalkvitteringer og state bevares.

105initial0 har ny eksakt PLAN/RECIPE/budget bundet til104/103/dc1c; bare
campaignklargjøring åpnes. Nytt frosset nativevindu, signert hardware og fysisk
Windowsboot kreves før32Entryforwards/0updates. training_enabled=false.
Initial/savedstate/paritet/Entry-onlyoptimizer/resume og bestilt epoke gjenstår. Den bestilte fulle epoken og målt
LONG/SHORT/FLAT-kvalitet/dekning/sporbarhet gjenstår. Ingen utvidelse til Exit,
TEST, broker, HPO eller andre modeller. Stabil langkjøring observeres15–30min.

DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Git eier historikken.
Metode og bruksanvisning: docs/NATIVE_LEARNING.md. Status står bare her og i
NEXT_RUN_POLICY; regler i GX1_RULES.md, automatisk lesing i gx1_handover.sh.
