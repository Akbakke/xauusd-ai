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

## Aktuell grense —095 initial0 ferdig;096 CPU-stateaudit klar

095 er faktisk ferdig med guardPASS/exit0:256 TRAIN +256 CONTROL-entries,
16 Entry-forwards per rolle og0optimizersteg. Initialt resultat målte
2589.611s; komponentoppsett867.906s og checkpoint0.508227s. Dette er ikke
treningsgjennomstrømning.182kildebindinger er fysisk hash-/statkontrollert
uendret ved0f857/b552. WindowsReady/LastResult0 og ingen bundne Windows-/
Linuxprosesser ble bekreftet; kun konsumert095task er deaktivert. Frysen er
løftet. Ingen095relansering, trening eller lærings-/edgekonklusjon. Alle registrerte
TRAIN-/CONTROL-utverdier, valg, mål-/maskedigest og aggregater er eksakt lik089;
bare to designstier per rolle skiftet. Native walltime er239.793s kortere
(8.48%), men dette er ingen kontrollert A/B eller målt treningsfart.

Fysisk Boot510/20.10 kl.01:38:51.500UTC er bekreftet med WindowsSSH/Ubuntu,
182kilder,27artefaktpayloads og11inputstatidentiteter bevart. Pending→archive
bevarte intent-SHA.095/REBOOT_CONFIRMED.json SHA
5f50df0c302bdf56fd79982e5478354df076fd65197862a67d3c0f1393b344a0.
Boot alene er ingen treningsadgang.

094 CPUfresh er COMPLETE121.926s/0forwards/updates/refits. Model/anchor/EMA
har samme9c4cf514… og initialstate-bytene sammec62cc1ad… som088.093 feilet
før modellkonstruksjon på en gammel testSHA-binding; den er konsumert og
bevart.094 retter kun den kvalifiserte bindingen, med b552-produksjonskilde.
096 gjenbruker eksisterende090stateaudit med én endret policykey; planen
binder faktisk094fresh/095initial0 og én4G/900s CPU-kjøring uten forwards.
Den må fysisk bevise model/anchor/optimizer/EMA/scheduler og CPU/Python/
NumPyRNG før streng eksisterende initialaudit godtas.

091 paritet feilet før optimizer på dynamic_terminal_net_markout_bps.
Original ACTUAL_FAILURE_REVIEW SHA
3ce95c60ec07c788706c200f386f7eafc898f6b82475d937567c75ee258b6c4c bevares.
Årsaken er fortsatt ubevist. Fire rene metadata-kall bruker nå eksisterende
finite file_verification_scope, lukket før restore/forward/update/optimizer.
Samme to paritetsforwards har CPU-only hostdispatch/max/RMS/failurediagnostikk;
ingen ny forward, kernelendring eller toleranseendring.092 kvalifiserte dette:
95PASS/3eldre eksplisitt uaktuelle SKIP,0fail/error. SOURCE_REVIEW SHA
612fd6ce722f8bb2f146607a6c1303972f9cb2f3f4d4e5f0836d5e0005fecca2.
Aktuell closure er b552bdb173050cbf7a8113ff71822ad6da049759cad57594cae68eaa391e3519.

En passiv095stack traff fortsatt uncached SHA under dynamiccohort-eieren
unified_exit_bounded_val_cohort_v1.py354–449. Den rene metadataoperasjonen
kan få ett eksisterende finite hashscope, med alle SHA/stat-/identitets-
kontroller beholdt og lukking før modellarbeid. Ingen endring ennå. Behold
b552 gjennom faktisk paritetsdiagnose; samle deretter denne målte I/O-retten
med eventuell målt paritetsrettelse for å unngå ekstra kildegenerasjoner.
Siste observerte prosessrchar02:28 var1.422TBlogisk, kernelregistrert lesing
25.158GB; dette er ikke totaltall eller1.422TBfysisk disklesing.

Neste:096stateaudit, deretter ny097teknisk paritetsdiagnose med samme kilde
og eksakte nye campaign-/boot-/maskinvareporter. Ingen full epoke før faktisk
paritet/Entry-only optimizer/resume er kvalifisert. training_enabled/fullEpoch/
fullVAL=false. Den bestilte epoken, full indikatorrevisjon og lærte LONG/SHORT/
FLAT-kvalitet/dekning/sporbarhet gjenstår. TEST/Exit/broker forblir stengt.

## PC, opprydding og metode

Brukerens18.10-omstart er utført;034/REBOOT_CONFIRMED.json er bindende.
Ikke utfør samme bestilling igjen. Seneste separate før-native Boot510 er
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
