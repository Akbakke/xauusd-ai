# Minimum native læring på D1-horisontmålet — forhåndsregistrert plan 12.10.2026

Status: **planlagt, ikke lansert.** Brukeren vedtok 12.10.2026 («ja gjør dette …
Bruk minimum nå i starten for å se at alt fungerer som det skal, så kan vi
vurdere å legge på fler ting senere»). Dette dokumentet binder hva «minimum»
betyr, i hvilken rekkefølge det bygges, og hva som ikke åpnes. Hvert steg får
egen policybinding, budsjett og kvittering; ingen steg autoriserer det neste.

## Hva minimum er, og hva det ikke er

Minimum er **omfang**, ikke features:

- Modellen er den fulle native v10-funksjonen på NATIVE_D1-generasjonens
  inputs (315 signalfelt med den niende D1-makro-/posisjoneringsfamilien, 132
  kontekstfelt, fem tidsrammer). Brukerens forslag om å sløyfe M5-sekvensen,
  candlestick-/mønsterfelt og M1-Exit er **ikke** tatt inn: regel 4 (alle
  genuine familier blir i den lærte stien), Multi-TF-kravet og Exit på M1 er
  bindende, og feature-kutting gir ingen målbar fart (ett forward på batch 16
  tar under ett sekund på CPU i nullstegsbaselinen). Åpning krever eget
  operatørvedtak skrevet inn i GX1_RULES.md.
- **Entry-først** (operatørvedtak 09.10): treningsfasen er `entry_only`; Exit-
  hodet beholdes i samme bundle, men Exit-tap og Exit-gradienter er inaktive.
  Dette er den rulekonforme formen av «sløyfe M1-exit nå».
- **Ett mål i tapet:** det observerte markedsmålet ved 20 handelsdager
  (`entry_observed_market_target_bps_h20d`, eksplisitt bundet etterfølger etter
  vedtakene 09.10 og 11.10). 1 og 5 handelsdager observeres på de samme radene
  og rapporteres, men optimeres ikke. Ingen horisont er en holdetidsregel.
- **Ett seed** (20260911), batch 16, samme optimizer/tap som den eksisterende
  Entry-only-fasen; ingen nye hyperparametre, ingen søk.
- **Avgrensede steg:** smoke (256 steg) → fit (1 024 steg på 256 rader, viser
  at tapet kan falle) → kurve (4 096 steg på faste TRAIN-rader, CONTROL4096
  observert ved 0/256/1 024/4 096 steg). Ingen full epoke, ingen full VAL,
  ingen adaptiv forlengelse, ingen seleksjon på CONTROL.

## Forutsetninger som finnes

NATIVE_D1-generasjonen (HISTORY2009W_NATIVE_D1_20261011_003) er bygget og målt
til og med fersk v10-nullstegsbaseline (run-root NATIVE_D1_REBUILD_20261011_003):
datasett med horisontmål i TRAIN/VAL, TEST forseglet, fysiske views, base- og
summary-normalisering, tensorparitet, fysisk indeks og økonomimetode. Den
eksisterende Entry-only-mekanismen (`entry_observed_market_v1`: læringsdesign,
datasettbinding, `require_entry_only_gradients`, læringsstudie-observatør) er
målt i bruk på V38-generasjonen (ENTRY_LEARNING_CURVE_20261010_001).

## Rekkefølge og budsjetter

**P0 — kode og tester (CPU, ingen kjøring).** Eier
`gx1/contracts/entry_observed_market_horizons_v1.py` får læringsdesignet for
horisontmålet (`require_entry_observed_market_horizons_scope`,
datasettbinding som leser de seks brutto-/elapsed-kolonnene én gang og emitterer
de eksakte batchnøklene for alle tre horisonter, batchmålvalg for
beslutningshorisonten 20 d). Treneren velger horisontmålet når recipen binder
designet; treningsobjektiv-kontrakten får fasen
`entry_only_observed_horizon_20d` med eksakt målbeskrivelse (ny SCHEMA_VERSION);
læringsstudie-observatøren rapporterer alle tre horisonter per sjekkpunkt
(rank-IC og MSE). Fokuserte tester på syntetiske fixtures beviser mekanikk,
aldri kvalitet. Spec: `configs/research/NATIVE_ENTRY_OBSERVED_HORIZONS_20261012.json`
(læringsdesign) og `configs/research/ENTRY_D1_HORIZON_LEARNING_STUDY_20261012.json`
(studie med faser, rader, observasjonssteg, gjennomgangsregel).

**P1 — CPU-forberedelse i run-root NATIVE_D1_REBUILD_20261011_003**, klonet fra
de aksepterte stegene med operators/clone_stage.py, ett tungt steg om gangen,
nytt budsjett PREPARATION_CPU_002:

1. NATIVE_PREPARATION_001: frosset kronologisk læringsdesign (TRAIN-vindu,
   train/control-kutt, CONTROL256, fysiske kilder) for den nye generasjonen.
2. SAMPLER_BENCHMARK_001: målt kapasitet for kandidatsamplerne på de nye
   inputs (315 felt); forrige generasjon målte 7 679 s per CPU-epoke ved
   batch 16. Målingen er obligatorisk fordi inputbredden er endret.
3. NATIVE_PREPROCESSING_001: fysisk mål-/normaliseringsbevis (LABELS_RESULT,
   NORMALIZATION_RESULT) for begge splitter.
4. NATIVE_COORDINATES_001: epoke-0-rekkefølge, TRAIN256-probe, CONTROL256 og
   TRAIN4096/CONTROL4096-koordinater uten utfallslesing.
5. NATIVE_METADATA_HANDOFF_001: valgt målt sampler og streng
   konstruktørmetadata.

**P2 — GPU gjennom eksisterende native campaign** (fersk fysisk Windows-boot
per invokasjon, eksisterende cgroup-/host-/GPU-/temperaturvakter, 12 000 s
vindu, to-slot-checkpoint hver 64. steg): smoke 256 steg → fit 1 024 steg →
kurve 4 096 steg. Hver fase er sin egen policybinding og campaign; brukeren
utfører omstartene. training_enabled settes bare for den bundne invokasjonen.

**P3 — gjennomgang.** Forhåndsregistrert regel fra læringsstudien 10.10
(seks parvise familiekorrigerte øvre konfidensgrenser under null mot initial og
TRAIN-konstant, kalenderuke-blokkbootstrap) på 20-dagersmålet, pluss
CONTROL4096 rank-IC per horisont per sjekkpunkt. Fit er TRAIN-tilpasning,
kurven er senere generalisering på gjenbrukt utviklingsdata; ingen av delene
er økonomi eller edge. Et kvalifisert resultat åpner bare neste vedtak
(flere seeds, lengre kurve, Entry-bevarende Exit-trening); et negativt
resultat lukkes ærlig.

## Hva planen ikke åpner

Ingen featurekassering, ingen ny arkitektur, ingen separat Exit-modell, ingen
TEST-lesing, ingen full epoke eller full VAL, ingen seleksjon på CONTROL, ingen
hyperparametersøk, ingen live/paper/broker/ordre/spending. GC-order-flow er
parkert. Sammenligningen mot D1-ridge-journalen på samme beslutningsdager
(steg 6 i NATIVE_D1_CONTEXT_HORIZON_DECISION_20261011.md) kommer etter P3 og
krever egen binding.
