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

## Aktuell grense —091 paritetsfeil, kun konkret retting tillatt

091 er FAILED/guardexit1,20.10 kl.00:14:15UTC. Feilen er
DYNAMIC_M5_TECHNICAL_PARITY_VALUES_CHANGED:dynamic_terminal_net_markout_bps
(full_train2352 fra2441). Den oppstod før tekniske optimizeroppdateringer.
Kvalifiseringsmappen finnes, men ingen resultatfil eller kanonisk native-
terminalkvittering ble produsert. Originale guard-/trainerlogger er bevart.
ACTUAL_FAILURE_REVIEW SHA3ce95c60ec07c788706c200f386f7eafc898f6b82475d937567c75ee258b6c4c.
Linux751 og Windows9156 er borte; taskReady/LastResult1 og0bundne prosesser
ble bekreftet00:26:35UTC. Kun konsumert091task er deaktivert. Frysen er løftet;
ingen091relansering. training_enabled/fullEpoch/fullVAL=false.

Fysisk Boot509/19.10 kl.22:29:44.500UTC er bekreftet med autentisert
WindowsSSH/Ubuntu,182kilder,26artefaktpayloadhasher og11inputstatidentiteter.
REBOOT_CONFIRMED SHA84676ef86a4767f86887f16b7b694df99d76057c92551ff248348a2a8246b34d.
En kanonisk pending→archive-flytting bevarte eksakt intent-SHA; den er eksplisitt
registrert. Boot alene gir ingen treningsadgang.

Målt oppsett1184.433s/invocation1335.183s; seneste prosessobservasjon00:04
viste2.994TBlogisk/20.859GBdiskregistrert lesing. En passiv stack traff uncached
hash under technicaladmission. Fire rene metadata-kall (opprinnelige824/1371/
1382/2383) bruker nå eksisterende finite file_verification_scope; scopene lukkes
før restore/forward/update/optimizer. Kildeendringen må fortsatt kvalifiseres.
Ingen kontroll, modellverdi eller toleranse er lempet.

Manglende paritetstall er også rettet i kilden: samme to forwards per rolle
registrerer faktisk hostdispatch med CPU-only profiler; max/RMS, antall avvik,
outputstørrelse og gjenoppretting følger PARITY_FAILURE.json ved feil. Ingen
ny forward, kernelendring eller toleranseendring. Hostdispatch er ikke CUDA-
kernel-/hastighetsbevis; overhead er umålt. Årsaken til091avviket er fortsatt
ubevist. 092 er bundet til én capped4G kilde-/syntetisk kontroll av finite scopes,
paritet og feildiagnose. Ingen ekte modelldata/updates/nativeinnrømmelse.

088fresh/089initial32forwards/090state-RNG-audit forblir faktiske bevis vedf571.
Ny full_train-byte krever ny closure/fresh/initial0/stateaudit gjennom eksisterende
gater; gammel091 blir aldri bevis for endret kilde. Den bestilte ene hele epoken,
full indikatorrevisjon og LONG/SHORT/FLAT-kvalitet/dekning/sporbarhet gjenstår.
Ingen læring/generaliserings-/økonomisk edge-konklusjon følger av denne feilen.

## PC, opprydding og metode

Brukerens18.10-omstart er utført;034/REBOOT_CONFIRMED.json er bindende.
Ikke utfør samme bestilling igjen. Seneste separate før-native Boot509 er
bekreftet med kvitteringen og bevaringsomfanget i aktuell grense ovenfor.
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
