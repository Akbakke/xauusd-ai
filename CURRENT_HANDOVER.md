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

## Aktuell grense — faktisk initialmåling083 fullført; neste faktiske treningssteg

083 fullførte initialbaseline med 256 TRAIN + 256 CONTROL, 16 Entry-forwards
per rolle, totalt32. Exit-forwards/optimizersteg/VAL-epoker=0. Canonical receipt
ble publisert19.10 kl.19:01:09UTC med guardPASS/exit0 og RESUMABLE, som er
forventet ved initial0taket. Dette er ingen fullført epoke eller målt læring.
Windows-controller5624 er avsluttet; taskDisabled kl.19:06:06UTC. Frysen er løftet.
Original083 er konsumert og relanseres aldri. Initialresultat:
/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_INITIAL_CONTROL_BOUND_20261019_083/.gx1-candidate-training-session.BUNDLE/initial_measurement/RESULT.json
SHA173ad22f59441cf797c3c6b328254a4f7c49e3d63c3ff801342a2f7d0eb60eda.
Terminalreceipt SHA4eee83d013f51e6e9fb7c0599c98708c96361c7035e2a0f9a2bb9ea58912beef.
ACTUAL_TERMINAL_REVIEW SHA24cb00f86dd12cd42b548be3e7ddff5b8bb8184e35ca4f03cfb540710422889b.

To uavhengige kilde-/outputkontroller fant konsistente082-init/055-normalisering/
057-mål/masker/koordinater og aktiv kompaktprofil. ONLINE=anchor beholder initial-
hash9c4cf514cbfd5b6f9366f69065abcc0878ef274117e2b5c7e703656e86a79b4a.
Separate komponentmasker og alle480 queries er bevart. Utrente diagnostiske
handlinger TRAIN LONG3/SHORT0/FLAT253 og CONTROL FLAT256 er baseline, ikke lært
selektivitet. Faktisk7lokale/5TF/25samarbeidstokens er observert; individuell
featurepåvirkning eller komplett formelkorrekthet følger ikke av dette.

083 samlet native4007.463s inkluderer setup1329.333s og checkpoint360.268s.
Ren encoderlatency er ikke separat målt; ingen trenings-ETA utledes av dette.
En enkelt ikke-blokkerende stackobservasjon målte gjentatt fysisk hashing i
ren dynamic-context-validering før modellarbeid. Metadatarettelse084 er
UTSATT/UKJØRT: dagens kontrakt ville ellers kreve ny CPUfresh og ny initial0,
selv med uendret modellmatematikk. For korteste korrekte vei til faktiske
treningssteg er productiontrainer restaurert byteidentisk310. Rettelse/test
bevares kun i Gitstash1678872c75485d7a0e1186d13c2a6ee53d894db3, ikke som aktiv
kode.084/DEFERRED_PATCH SHAd21f187b0b241661b5ba1e7f17aa0bad290f00e6ad25006d09a9a572d9809d02.
Ingen084test eller jobb ble startet; ingen nye PASS- eller hastighetsbevis.

Gjenbruk076TRAIN/078VAL-CONTROL/081199PASS/082CPUfresh/083initialbevis innen sine
uendrede roller. Neste steg er faktisk083initialaudit og ny eksakt privat4-
Entry-kvalifisering:2direkte TRAIN-steg mot1steg→durable load→1steg, totalt4
private updates og4paritetsforwards. Rekkefølge/state/RNG/optimizer/scheduler/
EMA/progress må samsvare; ingen Exit/CONTROL-updates. Hovedtreningsvekter
starter fortsatt fresh; privatkontrollvekter brukes ikke til hovedtrening.
training_enabled=false inntil denne avgrensede kvalifiseringen er eksakt bundet.
085 er nå bundet som én cappedCPUaudit av faktiske082/083checkpointbytes og
lagret modell/optimizer/EMA/scheduler/CPU-Python-NumPy-RNG. Den bruker ingen
modellforwards eller oppdateringer; strict initialaudit publiseres først etter
faktisk bevaringskontroll. PLAN ligger i
/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_INITIAL_STATE_AUDIT_20261019_085/PLAN.json
SHA69fa89bae94ce500c21ea5b6afd5d1d21eebb44799e9aff21de1262454a83504.
Deretter kreves eget opptak for den allerede bestilte ene nativeepoken; ny
campaign og gjeldende maskinvare-/bootporter beholdes. Individuelle formel-/
enhets-/warmup-referanser, komplette makrovintager og LONG/SHORT/FLAT-kvalitet/
dekning gjenstår. Ingen treningshastighet eller økonomisk effekt er målt ennå.
Generalisering og økonomisk edge er egne senere vurderinger.

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
