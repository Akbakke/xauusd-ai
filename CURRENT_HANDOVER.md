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
TEST fra juli er forseglet. Én full nativeepoke er bestilt og gjenstår. Ny108plan gjelder118249TRAIN-rader,
7391batcher (siste9), én epoke og ONLINE. training_enabled=true/fullEpoch=true
åpner bare denne eksakte planen etter nye campaign-/boot-/maskinvareporter;
fullVAL=false. Full trening er ennå ikke startet.
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

## Aktuell grense —107 native mekanikk PASS;108 er én hel epoke

107root: `/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_PARITY_RESUME4_EPOCH_GUARD_20261020_107`.
Faktisk RESULT SHA6379acd001a3a79889e518406bbc76b2a64e7bb2642f4652e401fd47633546f5.
Uavhengig ACTUAL_TECHNICAL_TERMINAL_REVIEW SHA373a730ba3220606209d5cac018bce2cf560ae7a692afcc9fcf08834bed91f5c.
Canonical receipt SHA275abc113c994a6fe096f4431ce2ed3178158d621dc4121e24c124c015b31492:
guardPASS/trainer0/observer0. Slutt12:07:58UTC; ingen bundne Linux-/Windowsjobber,
Windows-task Disabled/LastResult0.107er konsumert og skal aldri relanseres.

Fire private Entry-oppdateringer: uavbrutt2 mot1+durable reload+1, logiskcursor2.
Åtte state-trær er eksaktlike: modell/anker/optimizer/EMA/scheduler/RNG/epochorder/
trainingprogress.16TRAIN+16CONTROL ga fire parityforwards med åtte outputflater
per rolle: max/RMS-avvik0 ved beholdt atol1e-4/rtol0; masker/actions/ties er like.
0Exit/CONTROL-oppdateringer og ingen teacherrefresh. Dette er mekanikk, ikke læring.
Private107vekter er bare auditbevis og skal aldri brukes i hovedepoken.

182kilderoller/181filer og11inputstatidentiteter er bevart;26prebootkontroller
uendret, ett rebootintent konsumert av canonicaleier. POST_NATIVE_SOURCE_PRESERVATION
SHAd66037f58fc62d819b6ba1ce3f7aa296d10a6fc285c3a9a4101337d5767e05d0.
Aktuell funksjonsclosure dc1c6b627a2d42a9fea18fcd5dd6bdcfca493a840f32bbf01e1a765a919b785d
beholdes gjennom første helepoke. Ingen ny modell-/indikatorendring eller rebuild.

108root: `/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_SINGLE_FULL_NATIVE_EPOCH_20261020_108`.
Bruk eksisterende `entry_m5_path_single_epoch_run`-eier.118249TRAIN-rader brukes
nøyaktig én gang:7391batcher/batch16/partial9/drop_last=false. Tre forhåndsbundne
12000s-vinduer innen13800s-guard kan bare fortsette samme epoke/cursor til7391;
ingen automatisk ekstraepoke, utvidelse, Exit, fullVAL, TEST eller refit.
Oppskrift bygger på faktisk105/RECIPE_BOUND, ikke det inaktive105draftet.
Fjern initial0scope; bind strict106audit som chronological_learning_measurement
og behold057WITNESS.104fresh gjenopprettes via105/106-kjeden. Aktuelle bytes/stat
og uendret funksjonsclosure gjenbrukes; ingen ny fresh-/initial0runde uten endring.

Etter durable7391 kreves fast ONLINE-sluttmåling mot105baseline på TRAIN/CONTROL
og læringsreview. `complete=false` ved denne TRAIN-grensen betyr ikke at radenes
epoke er ufullført; faktisk scope/witness/finalmåling avgjør. Nådd7391åpner aldri
ny epoke. Kvalitet/dekning/sporbarhet og generalisering er fremdeles ubevist.
Eksisterende begrensning: sluttmåling må ferdigstilles før invokasjon+11940s;
train-wallpause er12000s, og en delvis sluttmåling har ingen automatisk resume.
Dette er en betinget kildebegrensning, ikke observert108feil; ingen forebyggende
kilde-/baseline-syklus nå. Bevar faktisk terminal/checkpoint ved et eventuelt stopp.

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

Seneste faktiske boot515/20.10kl.11:06:40.500UTC er bekreftet i107/REBOOT_CONFIRMED,
SHAf55ee8f5dbb5fa76c1ecff4fec6a36888d0fa489ce5aa2f849523ca2656fd438.
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
107målte første16raders batch til0.719scompute+0.217sloader (warmup). Senere
1batch-intervaller var0.594/0.727s; ingen full-epochETA utledes av fire oppdateringer.

Gjenbruk ferdige inputs/targets/055normalisering og kvalifisert kilde. Neste er
108eksakt singleepochcampaign, ny boot/maskinvareadgang og den ene hele epoken
etter faktisk admission. Kvalitet/dekning/sporbarhet, individuelt numerisk
featurebevis, generalisering og økonomisk edge gjenstår. Stabil jobb observeres
15–30min. Ingen Exit, TEST, broker, HPO eller nye modeller i denne bølgen.

DATA/RUNS ryddes bare av retention-eieren. Ingen Pythonfil er bevist slettbar
ved full referanse-/import-/CLI-/test-/eierkontroll. Git eier historikken.
Metode/bruksanvisning: docs/NATIVE_LEARNING.md. Status står bare her og i
NEXT_RUN_POLICY; regler i GX1_RULES.md, automatisk lesing i gx1_handover.sh.
