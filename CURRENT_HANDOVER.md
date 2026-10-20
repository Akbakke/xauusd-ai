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
faktiske profilbundne porter. training_enabled=true er kun for107s fire private Entry-oppdateringer;
fullEpoch/fullVAL=false. Full trening er ikke startet.
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
-103rettelsen bestod148fokuserte caser/0skip/0feil. Bare verifisert dynamisk
Entry-fase hopper over ubundet Exit-epochsetter; legacy forblir strict. Doble
policykontroller deler et kort hashscope uten å fjerne semantiske vakter.
-104genuinCPUfresh er COMPLETE/exit0,120.688s,0forwards/updates/refits og ingen
arvede vekter. ONLINE/anker/EMA-state9c4cf514… og initialblobc62cc1ad… bevart.
-105faktiskinitial0 er COMPLETE:256TRAIN+256CONTROL,32Entryforwards,0Exit,
0updates/epoker. Native1864.477s; guard/observer exit0. Original initialtilstand
9c4cf514… og alle komponent-/horisontmasker er bevart.182kilders SHA/stat og
11inputidentiteter bestod etterkontroll. Dette er ingen læring.

## Aktuell grense —106 savedstate bestått;107 native mekanikk er neste

105root: `/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_INITIAL_EPOCH_GUARD_20261020_105`.
ACTUAL_TERMINAL_REVIEW SHA96017e454d2e6a1a49219911d4b216abdb83df23a3e0f988d77c9511d56f8cb0.
Native receipt dffffae7d1405ed5b3a0fd1ee3be8d5a2b5711ce8a5fc1797588cca0a4241654.
105er konsumert og Windows-task Disabled/LastResult0; ingen bundne Linuxjobber.
Aktiv105recipe er **RECIPE_BOUND.json**, SHA3dfe8322…; RECIPE.json og gammelt
budsjett er bevarte inaktive drafts og skal aldri brukes til neste binding.

106root: `/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_INITIAL_STATE_AUDIT_20261020_106`.
Faktisk capped CPU-audit COMPLETE/exit0 på67.647s; direkte controllerwait exit0.
104fresh og105privat checkpoint er fysisk hash-/CPUload-kontrollert. Modell,
anker, optimizer, EMA, scheduler og lagret CPU/Python/NumPy RNG er eksakt
bevart. CUDA-RNG finnes, men sammenlignes ikke mot CPU-initialiseringen.
Ingen modellbygging/forwards/updates/refit/Exit/TEST. TERMINAL SHA9cb9bedc…,
INITIAL_MEASUREMENT_AUDIT SHA44cb6ec6…, STATE_PRESERVATION_PROOF SHA54a45a7a….
106er konsumert; ingen relaunch eller ny stateblob-lesing uten konkret behov.

Aktuell funksjonsclosure dc1c6b627a2d42a9fea18fcd5dd6bdcfca493a840f32bbf01e1a765a919b785d
beholdes gjennom107. Ingen ny modell-/indikatorendring eller datasett-rebuild.
107root: `/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_PARITY_RESUME4_EPOCH_GUARD_20261020_107`.
Ny PLAN og strict31felt TECHNICAL_QUALIFICATION_PLAN binder faktisk104/105/106.
Oppskrift bygges fra faktisk105RECIPE_BOUND og først etter gjeldende policy-
commit, slik at source_commit er eksakt.102controller gjenbrukes med kun stagekey
endret. Ny fysisk boot etter dagens514 og eksisterende signed hardware-porter
kreves; materialisering er ikke nativeadgang. Ingen claim eller nativejobb ennå.

107skal sammenligne no_grad/grad på16TRAIN+16CONTROL: fire parityforwards.
Deretter uavbrutt2steg mot fersk restore og1steg+durable reload+1steg:
fire faktiske private optimizeroppdateringer/fire trainingforwards og logisk
cursor2. Åtte saved-state-trær skal være eksakt like. CONTROL får ingen
oppdateringer, Exit/anker kjøres ikke og teacher refresh er av. Private vekter
skal aldri brukes som hovedepokens start. Full epoke/fullVAL er fortsatt stengt.

102s Exit-lifecycle-feilstopp er bevart og konsumert;103retter feilen.
104/105/106har kvalifisert oppstart og bevart tilstand, ikke current optimizer/
resume eller læring. Etter faktisk107PASS gjenstår eksakt admission og den
bestilte ene fulle nativeepoken. LONG/SHORT/FLAT-kvalitet og selektivitet er ubevist.

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

Seneste faktiske boot514/20.10 kl.09:19:51.500UTC er bekreftet i105/REBOOT_CONFIRMED,
SHAace2823f9e2e667b72fe55cadc70bdfda9bbed2c3589cc621ba1057ab083e09a.
Denne og18.10-bestillingen er fullført; ingen gjentakelse. Ny nativecampaign
følger egne ferske boot-/hostporter. Faktisk105profil300W/drawstop310W,
core85°C/memory80°C/VRAM12288MiB; source/hardware-owner/signedguard eier grensene.

MTF V4-NPY gjenbrukes read-only; reelle M5-sekvenser rekonstrueres uten tett
seq-memmap. Oppstart bygger fortsatt midlertidig split- og M5-signalbacking.
105/EFFICIENCY_SOURCE_REVIEW_002.json dokumenterer at én sekvenskaller bare
beholder signal, mens felles loader også skriver ctx/cat. Signal-only retention
er en uimplementert senere I/O-kandidat med alle verdi-/domene-/alias-/maskevakter
beholdt. Kumulativ rchar er logiske lesinger; Linux storage-I/O er ikke hosttotal
eller per-fil tidsprofil. Ingen kontrollert fartsgevinst eller trenings-ETA er bevist.

Gjenbruk ferdige inputs/targets/055normalisering og kvalifisert kilde. Neste er
107native Entry-only paritet/optimizer/resume og bestilt full epoke
etter faktisk admission. Kvalitet/dekning/sporbarhet, individuelt numerisk
featurebevis, generalisering og økonomisk edge gjenstår. Stabil jobb observeres
15–30min. Ingen Exit, TEST, broker, HPO eller nye modeller i denne bølgen.

DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Git eier historikken.
Metode/bruksanvisning: docs/NATIVE_LEARNING.md. Status står bare her og i
NEXT_RUN_POLICY; regler i GX1_RULES.md, automatisk lesing i gx1_handover.sh.
