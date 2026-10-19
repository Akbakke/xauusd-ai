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
Behold lukket M5/M15/H1/H4/D1 og de rikeste 26 månedene med GC MBP-1.
Kvalitet og dekning skal måles for både valgte og avståtte entries.

Første profil er `compact_mbp_smc_ema_atr_clock_macro_entry_v1`: MBP-1,
SMC/struktur, EMA/ATR og nødvendig klokke-/makrokontekst. Utvalgte candle-/
chart-pattern- og overlappende felt er parkert. Hele biblioteket og
fullmodellens åtte familier med niende D1/makrokontekst beholdes. Ingen felt
er erklært støy uten måling. Separate SMC/MBP-modeller, ensemble og M30
krever senere samme kronologiske OOF-evidens for merverdi.

TRAIN er [2024-05-01,2026-01-01), VAL [2026-01-01,2026-04-01), senere
utviklingskontroll [2026-04-01,2026-07-01). Alle 26 måneder er gjenbrukt
utvikling; TEST fra juli er forseglet. Én full nativeepoke er tidligere
bestilt og gjenstår etter faktiske profilbundne porter. training_enabled=false.
Ingen broker/live/paper/ordre/spending/promotion. De nøyaktig to godkjente
practice-leseoppslagene er konsumert; ingen gjenværende brokeradgang.

## Ferdig — gjenbruk eksisterende inputs og beståtte kontroller

- 66 GC-råfiler med 892100712 records og 9173893 prisnivårader er bevart.
  Identiske genuine T-poster dedupliseres ikke. 17 PTP-minutter/18 bok-event-
  bøtter sensureres. MBP-1 viser BBO/ordercount, ikke skjult dybde/ordreidentitet.
- 1205 GC-verdier med 1205 separate masker er publisert. Full fysisk flate er
  signal 2725/ctx-cont 2542/ctx-cat 1 og 253 felt per TF. 051-kilderekonstruksjon
  og 053-input-liveness bestod på 118249 TRAIN-/17284 VAL-rader. Ingen ny rebuild.
- 055 whole-TRAIN Entry-only-normalisering bestod; ingen Exit/VAL/TEST-fit.
  NORMALIZATION SHA c02ca97967a314f7dbd6c8bebc341c9782acd9e999e67e4b730db7e99e318c62.
  Hele transformen og domene-/finite-/alias-/maskevakter beholdes før gather.
- 057 binder faktisk native design, mål, koordinater og frosne 256 TRAIN-/256
  CONTROL-rader valgt uten utfallsutvalg. 118249 TRAIN-rader gir 7391 batcher,
  siste batch 9. 059 binder prospektive kostvilkår, ikke historisk kostsannhet.
- Kompaktprofilen beholder alle 1205 GC-verdier + masker og 61 D1/makrofelt.
  Aktiv flate er 2622 signal/2525 ctx/139 MTF. Modellen konstruerer og kjører
  bare aktive ruter: lokale encodere 8→7 og MTF familie×TF-ruter 40→25.
  Candle-ruten beholder 74 ekte MBP-kanaler. Lokal 96-bar-sekvens, MTF-vinduer
  16/64/96/96/252 og lagdybde beholdes. Fysisk I/O er fortsatt full flate.
- 061-profilkvalifisering: 380 unike kilde-/syntetiske caser bestod.
  065-dynamiske kildekontroller: 707 PASS, tre eksplisitt uaktuelle legacy-
  kombinasjoner SKIP; alle 71 nye caser bestod. Ingen ekte modelldata-forward.
- 068 publiserte faktisk CPU-freshstate på 268.828 s mot 066s 1285.314 s:
  4.781× raskere klargjøring, ikke målt modell-/treningsfart. 182 kildebindinger,
  tom optimizer og streng CPU-roundtrip bestod; ingen forward/steg/refit.
  Disse gamle kildebindingene er nå avløst av endret window/full_train-kilde;
  bevar originalen, men lag ny kildebundet freshstate før ny modellmåling.
- 070: 16 pytest PASS og faktisk Windows-hardening PASS. Metadatafrist 900 s;
  native-/ytre tidsgrense, task, temperatur-, effekt- og minnevakter er uendret.

## Aktuell grense — 083initial +085bevaringsaudit fullført;086I/O-rettelse

083 fullførte faktisk initialbaseline:256 TRAIN +256 CONTROL,16 Entry-forwards
per rolle, totalt32. Exit-forwards/optimizersteg/VAL-epoker=0. Canonical receipt
19.10 kl.19:01:09UTC er guardPASS/exit0/RESUMABLE ved initial0taket.
Original083 er konsumert; Windows-controller5624 avsluttet og taskDisabled.
RESULT SHA173ad22f59441cf797c3c6b328254a4f7c49e3d63c3ff801342a2f7d0eb60eda;
terminalreceipt SHA4eee83d013f51e6e9fb7c0599c98708c96361c7035e2a0f9a2bb9ea58912beef.
To uavhengige metadata-/outputkontroller bekreftet082-init/055-normalisering/
057-targets og aktiv kompaktprofil. ONLINE=anchor har initialhash9c4cf514...a79b4a.
Separate masker og480queries beholdes; faktisk7lokale/5TF/25samarbeidstokens er
observert. Utrente TRAIN LONG3/SHORT0/FLAT253 og CONTROL FLAT256 er baseline,
ikke lært selektivitet eller dokumentert featurepåvirkning/formelkorrekthet.

085 er faktisk COMPLETE/exit0/uendret81cd208c på63.040s, kl.19:21:18UTC.
Actual082fresh og083checkpoint er fysiskSHA-verifisert og CPU-lastet; modell,
targetmodell, optimizer, EMA, scheduler og lagret CPU/Python/NumPy-RNG er
sammenlignet eksakt. CUDA-RNG sammenlignes ikke mot CPU-initialiseringen.
Eksisterende strict initialaudit-eier bestod. Ingen konstruksjon/forwards/steg.
/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_INITIAL_STATE_AUDIT_20261019_085/TERMINAL.json
SHAb594d90907e7f903e93ac003331c93577b29040eb1f8697393aa7cdf5a080971;
INITIAL_MEASUREMENT_AUDIT SHA3ae5434c65736c2e1d2c85ebf863d858fad1f164d4511bc7d91b6d6c917bb07d.
Original085 er konsumert; launchPID133675 er borte, frysen er løftet.

Brukeren krever minst mulig unødvendig datalesing.086 retter konkret kildefunn:
checkpointprogress validerer policy gjentatte ganger via wholeTRAIN-SHA og
observed-scope/4096pointeraudit. Components/Dataset/M1provider gjenbrukes allerede
per nativeinvokasjon og privat4armer; ingen helfilhash/nyM1lasting/refit ble funnet
perbatch. Kallantall er kildeavledet, ikke målt fysisk disklesing eller tidsgevinst.
083samlet4007.463s inkluderer setup1329.333s/checkpoint360.268s; ren latency ukjent.

Minste086rettelse: Sessionlokalt, ikke serialisert bevis for eksakt policy/
prefix/monitor/kontrakt etter én streng fysisk+semantisk kontroll og full
filidentitetssjekk. Checkpoint/progress gjenbruker beviset med ny stat/type/
symlink/ctime-kontroll, uten wholeTRAINstream/pointeraudit. Ny Session/resume/
read-only validerer fysisk på nytt. Begge uavhengige kosthashstier delegerer til
felles eier, så alle kostavhengigheter inngår. Mutable checkpointSHA/load/save,
modell-/optimizer-/RNG-semantikk og alle target-/masker-/MTF-/profilregler består.
Den tidligere084purecontext-metadatarettelsen tas inn i samme nye kildebølge.
PLAN: /home/andre2/GX1_RUNS/NATIVE_M5_VERIFIED_CHECKPOINT_IO_20261019_086/PLAN.json
SHA86fd30a7901ffb5b6c764e4b9f4bca8b27516c4e9922e3123820307adf23ec7c.
087 er faktisk PASS_SOURCE_SYNTHETIC_ONLY på109.725s, kl.19:43:24UTC,
exit0/uendret81cd208c.375nåværende tester og21byteidentiske bevarte086PASS gir
396unikePASS, inkludert57nye;0feil/error/SKIP. Første086forsøk stoppet korrekt
på3raders nytestfixture; barefixture ble utvidettil17genuine TRAIN/VAL-rader,
medsamme strenge eiere/to batcher/halerad1. Original086feilkvitteringbevares.
Nyeguardtester måler råfilstreamgjenbruk/ingennyrålesing vedcheckpointmetadata,
fil-/kost-/NPY-/Parquet-/symlink-/ctime-/SHA-/prefix-/policy-/kontraktvakter,
ferskSession/resume/read-only, mutablecheckpointtamper ogscopegrense vedforward.
Eksisterende policy/session/coordinator/initial/kostregresjoner:339PASS.
SOURCE_REVIEW: /home/andre2/GX1_RUNS/NATIVE_M5_VERIFIED_CHECKPOINT_IO_QUAL_20261019_087/SOURCE_REVIEW.json
SHA64b1cbc7b2a4a27fa7718258acfd92601bda54e08bcede87a4e29c0c5c9d72ae.
TERMINAL SHA3f1fa64f01aa4e813b7bbf936b33fdf2b3887a0336fba9df9fc85fc77b52b12f.
Fysisk182-rollekontroll/181filer: barede5deklarerte metadataeierne erendret;
øvrigeclosurefeltereksakte. AST beviseropprinnelig modell-/state-/RNG-semantikk
etterfjerningavnyecontextargumenter; ingenmål/feature/normaliseringsendring.
Faktisknative disklesing/encoder-/treningshastighet er fortsattumålt.
training_enabled=false. Ingen source/synthetictest er native læringsadgang.

Endret086sourceclosure krever ny korrekt kildebundet CPUfresh/initialmåling og
nativeadmission.083/085 bevares som sine originale kildebevis. Deretter kreves
privat4 Entry-only updates/4paritetsforwards/direct-resume-ekvivalens, og eget
opptak for den bestilte ene hele nativeepoken. Private kontrollvekter brukes
ikke som hovedtrening. Alle individuelle genuine indikatorformler/enheter/
warmup på5TF, komplette makrovintager, LONG/SHORT/FLAT-kvalitet/dekning og
beslutningssporbarhet gjenstår i sitt fulle omfang. Ingen læring/generaliserings-
eller økonomisk edge-konklusjon følger av initialmålingen.

## PC, opprydding og metode

Brukerens18.10-omstart er utført;034/REBOOT_CONFIRMED.json er bindende.
Ikke utfør samme bestilling igjen. Seneste separate før-native Boot507 ble
bekreftet19.10 kl.17:45:21.500UTC med WindowsSSH/Ubuntu og182uendrede kilder.
083/REBOOT_CONFIRMED SHA0b37fe28a4d30d1d384a19b69e59d919204617a13a7d1ad56ac2badf70218039.
Dette er ingen treningsadgang. Ny campaign følger egne ferske maskinvare-/
bootkrav. Bruk kanonisk hasheier; campaign-JSON inkluderer avsluttende newline.

DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Historikk finnes i Git;
ingen kvitteringer/data/checkpoints slettes ved denne dokumentoppdateringen.
Brukerkrav19.10: prioriter effektiv datalesing og trening. Gjenbruk ferdige
inputs og skill engangsvalidering fra batcharbeid. Mål metadata/setup, batch-I/O,
GPU-steg og checkpoint separat; rchar er ikke fysisk disklesing. Se bindende
effektivitetsregel i GX1_RULES.md og training_efficiency_request_20261019 i policy.
Metode: docs/NATIVE_LEARNING.md. Feltbevis: docs/FEATURE_AUDIT.md.
