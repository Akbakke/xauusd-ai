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

## Aktuell grense — 073 fullført, ny 074 initialmåling skal klargjøres

071 er konsumert og ble kontrollert stoppet 11:08:48 UTC før modellstart.
Preflight gjenhashet samme TRAIN-fil: 427217476879 rchar, men bare 58176
physical-read-bytes. Dette var hovedsakelig filkontroll via sidebuffer, ikke
427 GB fysisk disklesing. Ingen BUNDLE/guard/progress/forward/optimizer.
Windows PID 8308 er avsluttet, task Disabled; Linux PG 576 har ingen levende
medlemmer. Original ACTIVE_INVOCATION og terminalkvitteringer bevares.
STOP_TERMINAL SHA 3a7a94257bf7c4c38989b7d20072ad7dd2ac65254f78933dbe42c8aff8604c1c.
071 skal aldri relanseres.

072 retter bare samme målte kontrollklasse i eksisterende handover-, native-
window- og full_train-preflight: én eksplisitt filverifiseringsoperasjon med
kvalifisert hashgjenbruk. Alle opprinnelige kontroller er bevart; scope lukkes
før budsjettpublisering eller modell-/statearbeid. Ingen modell-, feature-,
loss-, optimizer- eller normaliseringsmatematikk er endret.

072 er avsluttet med 294 unike PASS, inkludert alle 13 nye integrasjonscaser.
Original 293 PASS/1 FAIL er bevart. Feilen var en foreldet policy-terminal-
kobling; bare den koblingen ble rettet og den ene berørte casen kjørt på ny.
Alle eksisterende testfunksjoner er AST-identiske. FINAL_RECEIPT SHA
4a59db7b3eb021c351769bfce21262fe208f25d571275e131c7a7f8afdb340ad;
SOURCE_REVIEW SHA 3da28f7c75b14ab7dff72e53344c4d83230985955095d2494a90de88a99061b1.
Artefakter: /home/andre2/GX1_RUNS/NATIVE_M5_NATIVE_PREFLIGHT_HASH_SCOPE_20261019_072.
Faktisk storfil-preflight-tid etter rettelsen er ennå ikke målt.

## Neste nødvendige steg og ærlige grenser

1. 073 fullførte faktisk 11:38:42 UTC på 266.187 s, exit0/uendret 73f260bc.
   TERMINAL SHA ff4ff9d512907cfe487e8344c5f52dbdae11d14f1340080d9ffec1e7c648221d.
   INITIALIZATION SHA 5a9e7f05585cc0f0e55e26ce3f1e623f767e1f4023c345dabe00069d94c855a1;
   MEASUREMENT SHA 44580d7ee1b9b47384d5e2d6cb14deca39379e531124d5aaeeda0df2be2e132d.
   Alle 182 kildebindinger/181 unike filer og statfelt er kontrollert. Closure
   SHA 188fdf14a738829f0d412a9194a5db539578e3a7fac1bf653b5393686a1adaea.
   Fersk ONLINE=anchor=EMA0, tom optimizer og streng CPU-roundtrip bestod.
   Ingen gamle vekter, modellforward, steg, refit, Exit, TEST eller broker.
   Behold original073; engangsplanen er konsumert og relanseres aldri.
2. Ny 074-plan binder 256 TRAIN + 256 CONTROL, 32 encoderforwards og 0 steg.
   Materialiser ny campaign fra073s eksakte constructor/observedPLAN/11files,
   kildeclosure, initialization/measurement og uendrede055/057/059-kilder.
   Runrot: /home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_INITIAL_PREFLIGHT_BOUND_20261019_074.
   Kun statusdokumenter endres; ingen Pythonendring eller ny CPUfresh er nødvendig.
   Faktisk prepared Boot504 er lest; ferske maskinvareporter og NY fysisk boot
   kreves etter materialisering før native. 070-controller/caps er uendret.
   Eldre069-timing er historisk, ikke målt074fart. Mål faktisk felt-/maske-/MTF-
   konsum, komponenter, latency/minne og actionspor i den nye initialmålingen.
3. Deretter separat faktisk paritet og Entry-only optimizer/resume med fire
   private tekniske updates og fire paritetsforwards, ingen CONTROL-updates
   eller gjenbruk av kvalifiseringsvekter. Først så den bestilte nativeepoken.

Ingen ekte aktuell modellforward, gradient/optimizersteg, initialbaseline eller
nativeepoke er fullført. Full individuell formel-/unit-/warmup-referanse,
makroens komplette originalvintager, læring, generalisering, økonomisk edge og
valgte/avståtte LONG/SHORT/FLAT-kvalitet er fortsatt ubevist. Gatevekter alene
forklarer ingen beslutning. Alltid FLAT er ingen dokumentert selektivitet.

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
