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

## Aktuell grense —100 initial0 bestått;101 tilstandskontroll klar

097 er konsumert FAILED/guardexit1 før optimizer:2TRAIN-forwards/0CONTROL,
maxterminalavvik0.000135153532bps mot låst1e-4. Original PARITY_FAILURE.json
og ACTUAL_FAILURE_REVIEW bevares. Ulike no_grad/grad-hostdispatch ble målt;
ingen toleranse, presisjon, features eller arkitektur endres for å passere.
097task er deaktivert; ingen relansering eller privatvektgjenbruk.

098 rettet bare de to målte blokkeringene: ett eksisterende finite
file_verification_scope rundt ren dynamiccohort-metadata, og midlertidig
MHAfastpathFalse rundt deterministisk dynamisk Entry-forward. Alle gamle
SHA/stat-/identitetskontroller bevares; scope lukkes før modell/optimizer.
MHAflagget gjenopprettes i finally; autograd, SDPAvalg, RNG, modellmodus,
presisjon og øvrig modell-/trainer-AST er uendret. Faktisk219PASS/3eksakte
kjente legacy-SKIP/0fail/error; alle25nye caser PASS.098 er konsumert.
Source_reviewSHAe92b41251f30821b8ef18747bc181dc6703cf522edeac728e4fa3d12ca284c22.
Ny closure8517c075eb74dacc85c21c2441d6e00d23335f0eff02144bc32b5e009e41924f
har182roller/181filer/7153839B. Bare kohortowner og trainer endret bytes.

099 full metadataprecheck206–259 inklusive bestod faktisk capped, inklusive
sen065kobling, uten falskCLAIM/modellimport. Deretter bestod én faktisk
4G/512M CPUfresh121.070133s,20.10 kl.04:42:36UTC.0forwards/updates/refits,
kilde/policy uendret og alle182kilders bytes/stat kontrollert. Hele
087→092→098kjeden er bevart. Terminal/result og ACTUAL_CPU_PREPARATION_REVIEW
ligger under NATIVE_M5_COMPACT_FRESH_COHORT_ATTENTION_20261020_099.
Constructorstate9c4cf514… og72MBinitialstatec62cc1ad… er eksakt lik094;
ny kilde/beregningsbane krever likevel fersk nativeinitialmåling.099 er
konsumert; ingen relansering. Worker rchar49.229GB er logiske lesebytes;
lagringsregnskap44.181GB er separat. Ingen påstand om samlet host-I/O eller
målt nativehastighetsgevinst. De ferdige datasettene og normaliseringen gjenbrukes.

100 er faktisk COMPLETE/guardPASS/exit0, konsumert20.10 kl.05:56:46UTC.
TRAIN256+CONTROL256 ga16+16Entry-forwards,0optimizersteg og0Exit-forwards.
Initialmålingen tok1869.792s; kanonisk invokasjon2049.761s. Komponentoppsett
739.453s og checkpointskriving0.539s. Forrige095initialmåling tok2589.611s;
dette er observerte rundetider, ikke kontrollert A/B eller treningshastighet.
Mål/kohorter/seks komponentmasker er bevart mot095; queryaksen er1..480.
Initiale handlinger TRAIN3LONG/0SHORT/253FLAT, CONTROL256FLAT er ikke lært
selektivitet. Modell/anker-statehash9c4cf514… er lik099constructor.
Alle182kilder SHA/stat,39statiske artefakter og11inputidentiteter er bevart.
Ingen rådatastrøm ble rehashet av sluttkontrollen.100task er deaktivert med
LastTaskResult0 og ingen bundne prosesser. ACTUAL_TERMINAL_REVIEW.json:
220a6360d25a3d6e57db92b6bd2ddeaac4ca01efc15f78df0542998fcd065b98.

Ny101 er bare én capped4G/512M CPUkontroll av faktisk099fresh og100lagret
initialtilstand. Eksakt modell/anker/optimizer/EMA/scheduler og lagret CPU/
Python/NumPyRNG skal sammenlignes med eksisterende streng eier. Ingen modell-
forward, update, refit eller nativecampaign. Etter101 kreves en ny eksakt
native teknisk4-plan for paritet/Entry-onlyoptimizer/resume før bestilt epoke.
100/095/096/097 er konsumert og skal ikke relanseres.

Seneste fysiske WindowsBoot512/20.10 kl.05:18:17.500UTC er bekreftet med
WindowsSSH/Ubuntu,182kilder/39artefakter/11inputidentiteter bevart.
100/REBOOT_CONFIRMED SHA e82a5f3a013a294af02221d755f05c0e398fb153a52f0097c5187005695674bf.
Dette er gjennomført100profilboot; neste nativecampaign har egne porter.

Individuelt formel-/enhets-/warmupbevis er supplert for350GCverdier med350
separate masker og23tekniske formler×5TF=115feltposter. De tre GCgruppene
215profil/trade+75bok/flow+60SMC er disjunkte. Manifest under
NATIVE_M5_READONLY_FORMULA_UNIT_WARMUP_20261020_001 harSHA
b4919b15710a957b34c406d04a9175963272c0748fa35f68f29ebddf93773911.
Dette er kildebevis;855andreGCverdier er ikke behandlet i denne bølgen,
og komplette uavhengige tallreferanser og individuelt modellkonsum gjenstår.
Bestilt epoke, full indikatorrevisjon og lært kvalitet/dekning/sporbarhet er
fortsatt uferdig. Ingen lærings-, generaliserings- eller økonomisk edge-konklusjon.
TEST/Exit/broker er stengt.

## PC, opprydding og metode

Brukerens18.10-omstart er utført;034/REBOOT_CONFIRMED.json er bindende.
Ikke utfør samme bestilling igjen. Seneste separate før-native Boot512 er
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
