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

## Aktuell grense — 074 feilet før modell, 075 retter tidsvindukontrollen

073 CPU-freshstate er fullført og konsumert; originale bindinger og state bevares.
074 materialiserte faktisk ny campaign, bekreftet fysisk Boot505 19.10 kl.
12:32:21.500 UTC, og startet én capped nativeinvokasjon kl.12:45:26 UTC.
Guard avsluttet child med status1 kl.13:13:22 UTC; Windowscontroller PID8568
var avsluttet kl.13:18:59. Task er nå Disabled, LastTaskResult1; ingen
relevante WSL-prosesser står igjen. 074 relanseres aldri.
Handover-terminalen binder faktisk exit1 og source_unchanged=null, fordi
full182filers postfailure-seal ikke ble utført før den eksplisitte rettelsen.

Faktisk feil er ENTRY_PHYSICAL_AUX_WINDOW_MISMATCH i TRAIN Dataset-binding,
før modellkonstruksjon: declared_window.end er lik korrekt eksklusiv cutoff,
men gammel kontroll krevde <cutoff. Manifest-/designklokker bruker også ulike
tekstformater for samme UTC-instant. 075 normaliserer klokkelikhet og bruker
samme dynamiske low<declared_end<=high som den eksisterende target-receipt-
eieren. Legacy <high, faktiske entryklokker <high, kilde-/cellebindinger og
uavhengige framtidsmasker består. Ingen modell-, mål-, taps- eller optimizer-
matematikk endres.

Normal canonical invocation-receipt mangler fordi record-eieren krever en
checkpoint-pointer også ved FAILED før modell. Ingen pointer/progress finnes;
original ACTIVE og logger bevares uten fabrikkert kvittering eller recordretry.
Separat faktisk feilobservasjon:
/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_INITIAL_PREFLIGHT_BOUND_20261019_074/FAILED_OBSERVATION_TERMINAL.json
SHA dc12ed3a44a3c99788603739307a67e969013aa1860a10b018c8e9e90bfb2557.

075: /home/andre2/GX1_RUNS/NATIVE_M5_PHYSICAL_WINDOW_BINDING_REPAIR_20261019_075.
075 avsluttet:178 PASS,0 FAIL/ERROR/SKIP, inkludert10 nye regresjonscaser.
Øvrig produksjons-AST og alle eksisterende testfunksjoner er identiske.
SOURCE_REVIEW SHA 395804febc28c1d67ceeb7d683121fb29ebf91e11473f3fe60002a5954d4820b.
Neste steg er fokuserte TRAIN/CONTROL-regresjoner og faktisk source-bundet
CPU Dataset-/målbinding før ny profilbundet freshstate/initialmåling. Ingen
ny native campaign er innrømmet; training_enabled=false. Gjentatte storfil-
kontroller er målt, men de er faste per datasett, ikke per rad/horisont.

Deretter gjenstår ekte kompaktmodellkonsum, initialbaseline/latency/minne,
separat faktisk paritet og Entry-only optimizer/resume med fire private
tekniske updates/fire paritetsforwards, og den bestilte nativeepoken etter
alle gater. Ingen ekte aktuell forward, optimizersteg eller epoke er fullført.
Individuell formel-/unit-/warmup-referanse, komplette makrovintager, læring,
generalisering, økonomisk edge og LONG/SHORT/FLAT-kvalitet er fortsatt ubevist.

## PC, opprydding og metode

Brukerens 18.10-omstart er utført; 034/REBOOT_CONFIRMED.json er bindende.
Ikke utfør samme bestilling igjen. Separat fysisk før-native-omstart for 071
ble bekreftet: Boot 504, 19.10 kl. 10:44:00.500 UTC, WindowsSSH/Ubuntu og
182 uendrede kildebindinger. 071/REBOOT_CONFIRMED SHA
a3e3597c4580f42eb9ab3f0089fbc91abd8ed375d4775af2b67f93aec93affcf.
Original confirm, faktisk abort og separat reschedule er bevart. Bruk alltid
kanonisk hasheier; campaign-JSON-hash inkluderer avsluttende newline.
Ny campaign følger egne ferske maskinvare-/bootkrav.

DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Historikk finnes i Git;
ingen kvitteringer/data/checkpoints slettes ved denne dokumentoppdateringen.
Metode: docs/NATIVE_LEARNING.md. Feltbevis: docs/FEATURE_AUDIT.md.
