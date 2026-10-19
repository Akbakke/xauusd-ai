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

## Aktuell grense — 080 feilet før måling; avgrenset 081-kilderettelse

080 er konsumert FAILED_NO_ADMISSION. Native exit1 kl.16:32:51UTC19.10:
CANDIDATE_PREFIX_CONTROL_BINDING_MISMATCH ved coordinatorens cutoffkontroll.
Datasett/modell ble konstruert, men initialmåling, session/checkpoint-pointer,
progress og canonical terminalreceipt ble ikke publisert. Ingen lærings-/
forwardsgevinst er bevist. Controller2540 sluttet exit1 kl.16:38:29UTC;
Windows-task Disabled kl.16:39:16UTC. OriginalACTIVE/logger bevares, ingen
pointer/progress/receipt er fabrikkert. Separat faktisk feilobservasjon:
/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_INITIAL_DATA_BINDING_BOUND_20261019_080/FAILED_OBSERVATION.json
SHA845723e19d68d61163e2aac6ed9293dabbcdabd1ed9ef26b88f3a9bf24282847.
Alle182kildebindinger/181unike filer/7138661bytes og statfelt ble kontrollert
mot fryst8ca998bd før081; kilde/policy var uendret under080. Ingen nativeprosesser.

Faktiske076TRAIN118249- og078VAL17284/CONTROL256-koblinger er bevart; de er
separate databevis, ikke coordinator- eller modellmåling. 079CPU-fresh fullførte
på265.962s uten forwards/updates/refit, med ONLINE=anchor=EMA0 og tom optimizer.
Original079 er konsumert. Endret081closure krever ny faktisk kildebundet fresh-
publisering; gamle vekter eller gamle initialstier kan ikke gjenbrukes som ny.

081 retter bare den observerte coordinator-feilen og målte metadatahashinger.
TRAINcutoff skal fortsatt være1.januar2026; CONTROL/VAL bruker1.april2026 fra
sammeSHA-bundne DESIGN.calendar, med eksakt rolle/split. Ingen targetmasker,
TRAIN-/TEST-tilgang eller legacy utvides. Liveness-digest bruker eksisterende
verifisert hasheier; egne metadata-scopes lukkes før Dataset/modell/restore.
Alle forventet-SHA/stat/innholds-/fullscan-kontroller består.199fokuserte tester
bestod, inkludert14nye; ingen feil/SKIP og uendret kilde under116.838s kjøring.
AST/syntaks/stale-path/diffkontroll bestod; eksisterende coordinator-testnoder
er AST-identiske. SOURCE_REVIEW SHA60ab082cce967296cb165f80d5b4d5c5a1050da6c60d12fc5c80ebd23e0e24b4.
081PLAN ligger i /home/andre2/GX1_RUNS/NATIVE_M5_CONTROL_CUTOFF_METADATA_REPAIR_20261019_081.
Ingen ny native campaign, modellmåling eller trening er innrømmet;
training_enabled=false. 074/076/079/080 og øvrige konsumerte planer relanseres aldri.

Neste: publiser ny faktisk kildebundet freshstate etter081 og bind en ny eksakt
initial0campaign med gjeldende maskinvare-/bootporter. Deretter kreves ekte
initialbaseline/konsum/latency/minne, separat faktisk paritet og Entry-only
optimizer/resume med4 private tekniske updates/4paritetsforwards, før den
bestilte ene nativeepoken. Individuelle formel-/enhets-/warmup-referanser,
komplette makrovintager, LONG/SHORT/FLAT-kvalitet/dekning og læring gjenstår.
Generalisering og økonomisk edge er egne senere vurderinger.

## PC, opprydding og metode

Brukerens18.10-omstart er utført;034/REBOOT_CONFIRMED.json er bindende.
Ikke utfør samme bestilling igjen. Seneste separate før-native Boot506 ble
bekreftet19.10 kl.15:09:14.500UTC med WindowsSSH/Ubuntu og182uendrede kilder.
080/REBOOT_CONFIRMED SHAeb380144e863ba44cff10d71cafa8bbb796f8e9d2020c426f0c5b48b5ab55f2d.
Dette er ingen treningsadgang. Ny campaign følger egne ferske maskinvare-/
bootkrav. Bruk kanonisk hasheier; campaign-JSON inkluderer avsluttende newline.

DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Historikk finnes i Git;
ingen kvitteringer/data/checkpoints slettes ved denne dokumentoppdateringen.
Metode: docs/NATIVE_LEARNING.md. Feltbevis: docs/FEATURE_AUDIT.md.
