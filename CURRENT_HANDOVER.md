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
bestilt og gjenstår etter faktiske profilbundne porter. training_enabled/fullEpoch/fullVAL=false;097teknisk plan er konsumert.
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

## Aktuell grense —097 målt paritetsfeil;098 minimal samlet rettelse

097 er FAILED/guardexit1,20.10 kl.03:56:24UTC, før optimizer. Faktisk diagnose
ble atomisk publisert i DYNAMIC_M5_TECHNICAL_QUALIFICATION/PARITY_FAILURE.json,
SHA bbc5fe4875dc1fcd732a933e1bf13b2f3e131cb3decd8fa59f3a69915f1bd811.
Det er målt2TRAIN-forwards,0CONTROL og0optimizeroppdateringer. Første felt,
dynamic_terminal_net_markout_bps, har max0.000135153532bps mot låst1e-4,
RMS0.000056246411bps og960celler over terskel. Senere felt er ikke kontrollert.

Begge passeringer brukte CUDA/eval/FP32 uten autocast, samme input/vekter og
gjenopprettet RNG/modulmodus/gradienttilstand. no_grad registrerte64hver av
fusedTransformerEncoder/nativeMHA; enable_grad registrerte0av disse. Begge
registrerte64SDPA-efficient-kall. Dette beviser ulike hostdispatch-veier og
det konkrete numeriske avviket; en rettelse er ennå ikke nativeverifisert.
Ingen toleranse, mål, features eller arkitektur skal endres for å passere.
Originale logger og diagnose bevares. Kanonisk native-terminalkvittering ble
ikke produsert; ACTUAL_FAILURE_REVIEW er bare etterprøvbar sammenstilling,
SHA634ff42d1443be2901807dfb72563abf57385e7eac5a06acda4f0f18e7107654.

WindowsReady/LastResult1/0bundne prosesser og Linux749 borte er bekreftet;
kun konsumert097task er deaktivert. Alle182kildebindinger og policy ble
kontrollert uendret ved b26aeedd/b552 før frysen ble løftet. training=false,
fullEpoch/fullVAL=false. Ingen097relansering eller gjenbruk av private vekter.

098 har implementert to målte blokkeringer i én kildegenerasjon: ett eksisterende
finite file_verification_scope i den rene dynamiccohort-eieren, og minste
rettelse som gir konsistent støttet MHA-beregningsvei mellom no_grad og grad
for deterministisk dynamisk Entry-only. Kohortscopet lukkes før modell/
optimizer/forward; SHA/stat-/identitetskontroller og legacy beholdes.
Root eier policy/launch/dokumentasjon, mbp_audit kohort+fokustester, og
paritetsagenten eier trainerens eksisterende _model_forward_fp32/presisjons-
context og fokustester. MHAfastpathFalse gjelder bare dynamisk Entry/
deterministiskFP32 rundt modellkallet og gjenoppretter tidligere flagg i finally;
SDPAvalg, autograd, modellmodus, RNG, presisjon og øvrig trainer-AST er uendret.
Hele kohort-AST er ellers identisk; eksisterende tester er uendret. Én098
capped kildesyntetisk kvalifisering er klar, med25nye caser. Ingen CUDA-/
hastighets-PASS er målt på denne kilden.
Ingen ekte modelldata-forward eller optimizer er autorisert i denne bølgen.
Nye kildebytes krever ny closure/CPUfresh/faktisk initial0/stateaudit og
paritet/Entry-only optimizer/resume gjennom eksisterende gater før epoken.
Ny099CPUfresh må bevare087→092→098-kvalifiseringskjeden med nøyaktig
de to098produksjonsstiene som overlay og to testfiler separat.087s direkte
trener-SHA-kontroll og092s after-closure må bindes til korrekt generasjon.
Prelaunch må gjennom hele publisher153–205, også065-koblingen171–178;
controllerhjelpere alene fanget ikke den faktiske093-feilen.

Gjenbruk095/096 som avgrenset b552-bevis:095 bestod32Entry-forwards,256TRAIN
+256CONTROL,0updates,2589.611s. Alle registrerte utverdier, valg, mål-/maske-
digester og aggregater er eksakt lik089; bare to designstier skiftet. Dette
var8.48%kortere initial-walltime, ingen kontrollert A/B eller treningsfart.
Komponentoppsett867.906s og checkpoint0.508227s.096 bestod59.802s med eksakt
model/anchor/optimizer/EMA/scheduler og CPU/Python/NumPyRNG bevart; auditSHA
aecb65ca1f17eb44803a39f33b3cead32221e8702cf01d466f1a985d68db2986.
094CPUfresh bestod121.926s/0forwards/updates/refits, med samme constructor-
state9c4cf514… og initialstate-bytec62cc1ad… som088. Ingen nye rådata bygges.

Fysisk Boot511/20.10 kl.03:24:10.500UTC er bekreftet med WindowsSSH/Ubuntu,
182kilder,38artefaktpayloads og11inputstatidentiteter bevart. Canonical
pending→reboot-intent-archive bevarte eksakt payload.097/REBOOT_CONFIRMED SHA
c16f593ad9e29f036d18f9e9ace92740705c466c5cb775a47b9182b4aa49d4e4.
Boot alene er ingen ny treningsadgang. Den bestilte epoken, full indikator-
revisjon og lærte LONG/SHORT/FLAT-kvalitet/dekning/sporbarhet gjenstår. Ingen
lærings-/generaliserings-/økonomisk edge-konklusjon. TEST/Exit/broker stengt.

## PC, opprydding og metode

Brukerens18.10-omstart er utført;034/REBOOT_CONFIRMED.json er bindende.
Ikke utfør samme bestilling igjen. Seneste separate før-native Boot511 er
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
