# Minimum native læring på D1-horisontmålet — forhåndsregistrert plan 12.10.2026

Status: **gjennomført og gjennomgått 14.10.2026 — REJECT_ENTRY_QUALIFICATION.** P0–P3 er
utført (policyblokk native_d1_minimum_learning_20261012, studiegjennomgang
`/home/andre2/GX1_RUNS/ENTRY_D1_HORIZON_LEARNING_20261012_003/STUDY_REVIEW.json`);
avvik fra ordlyden under er registrert i samme blokk (14.10). Brukeren vedtok 12.10.2026 («ja gjør dette …
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
- **Avgrensede steg:** fit (1 024 steg på 256 rader, viser at tapet kan falle;
  observasjonen ved steg 256 er planens «smoke», ingen egen smoke-fase finnes)
  → kurve (4 096 steg på 65 536 unike TRAIN-rader i frossen epoke-0-rekkefølge;
  TRAIN_PROBE4096 observert ved 0/256/1 024/4 096, CONTROL4096 ved 0 og 4 096
  slik studiekontrakten binder). Ingen full epoke, ingen full VAL, ingen adaptiv
  forlengelse, ingen seleksjon på CONTROL. (Ordlyd rettet 14.10.2026.)

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

Faktisk rekkefølge (14.10.2026): NATIVE_PREPARATION_003 (to feilede kloner
først, bevart) → SAMPLER_BENCHMARK_001 (stoppet: projisert over vinduet, lukket)
→ SAMPLER_BENCHMARK_002 (57 822 s, valgt 32 768/batch 16, målt epoke 7 906,5 s)
→ NATIVE_METADATA_HANDOFF_001 → NATIVE_PREPROCESSING_001 → NATIVE_COORDINATES_001.

1. NATIVE_PREPARATION: frosset kronologisk læringsdesign (TRAIN-vindu,
   train/control-kutt, CONTROL256, fysiske kilder) for den nye generasjonen.
2. SAMPLER_BENCHMARK: målt kapasitet for kandidatsamplerne på de nye
   inputs (315 felt); forrige generasjon målte 7 679 s per CPU-epoke ved
   batch 16. Målingen er obligatorisk fordi inputbredden er endret.
3. NATIVE_METADATA_HANDOFF: valgt målt sampler og streng konstruktørmetadata.
4. NATIVE_PREPROCESSING: fysisk mål-/normaliseringsbevis (LABELS_RESULT,
   NORMALIZATION_RESULT) for begge splitter.
5. NATIVE_COORDINATES: epoke-0-rekkefølge, TRAIN256-probe, CONTROL256 og
   TRAIN4096/CONTROL4096-koordinater uten utfallslesing.

**P2 — GPU gjennom eksisterende native campaign** (fersk fysisk Windows-boot
per invokasjon, eksisterende cgroup-/host-/GPU-/temperaturvakter, 12 000 s
vindu, to-slot-checkpoint hver 64. steg): fit 1 024 steg → kurve 4 096 steg.
Hver fase er sin egen policybinding og campaign; brukeren utfører omstartene.
training_enabled forblir false: studiekontrakten krever det, og fasene kjører
under studiens egen binding, ikke som native launch. (Rettet 14.10.2026.)

**P3 — gjennomgang.** Forhåndsregistrert regel fra læringsstudien 10.10
(seks parvise familiekorrigerte øvre konfidensgrenser under null mot initial og
TRAIN-konstant, kalenderuke-blokkbootstrap) på 20-dagersmålet, pluss
CONTROL4096 rank-IC per horisont per sjekkpunkt. Fit er TRAIN-tilpasning,
kurven er senere generalisering på gjenbrukt utviklingsdata; ingen av delene
er økonomi eller edge. Et kvalifisert resultat åpner bare neste vedtak
(flere seeds, lengre kurve, Entry-bevarende Exit-trening); et negativt
resultat lukkes ærlig.

**Resultat 14.10.2026 (målt på ekte deklarerte bytes):** fit memorerer
(kontrast-rank-IC 0,07 → 0,97 på TRAIN256). Kurven lærer de trente radene
(TRAIN_PROBE4096 kontrast-MSE 725 499 → 116 061, rank-IC 0,96) uten å
generalisere: CONTROL4096 kontrast-MSE 2 032 815 → 2 157 135 mot TRAIN-konstant
1 997 171, rank-IC 0,039 → 0,040 ved 20 d (0,001 → 0,085 ved 5 d), handlinger
FLAT 3 082 → LONG 2 893, valgt-handling-snitt +50 → −3 bps mot alltid-LONG
+148,5 bps. Alle seks forhåndsregistrerte parvise intervaller dekker null.
Studiegjennomgang v2: REJECT_ENTRY_QUALIFICATION. Dette er gjenbrukt
utviklings-CONTROL, ikke økonomi eller edge. Neste fase (brukerens ønske: et
mindre kronologisk vindu, f.eks. ett år, før «stor trening») krever eget
operatørvedtak og egen policybinding.

## Fase «recent_year» — forhåndsregistrert 14.10.2026 (vedtak 14.10 b)

Brukeren vil se et mindre kronologisk utgangspunkt før «stor trening». Fasen
bindes i etterfølger-spesifikasjonen
`configs/research/ENTRY_D1_HORIZON_LEARNING_STUDY_20261014.json` (hashbinder
forgjengeren; fit og kurve gjentas byte for byte) og i egen policybinding:

- **Populasjon:** de kvalifiserte TRAIN-radene hvis beslutningstid ligger i de
  siste 365 dagene før TRAIN-kuttet (2024-05-02T22:00Z … 2025-05-02T19:20Z):
  65 148 rader, målt fra den bundne TRAIN-klokken av kontrakteieren
  (`recent_window_rows`), aldri et literal. Samme frosne epoke-0-rekkefølge,
  begrenset til vinduet først; resten av rekkefølgen følger etter og nås aldri.
- **Budsjett (avledet):** optimizersteg = floor((65 148 − 1)/16) = 4 071, altså
  65 136 unike trente rader — praktisk talt samme antall som kurven (65 536),
  men konsentrert i siste år. Ingen epoke fullføres, ingen VAL-fase, én
  invokasjon (kurven målte 0,72 s/steg + 1 100 s oppsett).
- **Observasjon:** TRAIN-probe = de første 4 096 vindusradene i rekkefølgen ved
  0/256/1 024/4 071 steg; CONTROL4096 (uendret) ved 0 og 4 071. Samme seed,
  tap, batch, ferske vekter.
- **Regel:** samme seks parvise familiekorrigerte grenser på CONTROL4096 ved
  20 d mot initial og mot TRAIN-konstanten for de trente vindusradene; kurvens
  CONTROL-observasjon ved 4 096 rapporteres ved siden av, velges aldri. Hypotesen
  er at rader fra siste regime generaliserer til utviklings-CONTROL der hele
  historikken ikke gjorde det. Negativt resultat lukkes ærlig; positivt åpner
  bare neste vedtak (full epoke / flere seeds).

## Fase to-års-vindu — forhåndsregistrert 14.10.2026 (operatør: «kjør»)

Etter recent_year-resultatet (REJECT med retningssignal: CONTROL kontrast-MSE
2 032 815 → 1 877 306, rank-IC 20 d 0,04 → 0,19, valgt handling +50 → +128 bps,
alle seks intervaller dekker null) vedtok brukeren å kjøre det presenterte
alternativet 1: samme fase med 730 dagers vindu, bundet i
`configs/research/ENTRY_D1_HORIZON_LEARNING_STUDY_20261015.json` (hashbinder
12.10-forgjengeren; fit og kurve gjentas byte for byte; kontraktnøkkelen heter
fortsatt recent_year, window_days bærer sannheten):

- **Populasjon:** kvalifiserte TRAIN-rader med beslutningstid i de siste 730
  dagene før kuttet: 128 815 rader → 8 050 steg (avledet av kontrakteieren),
  128 800 unike trente rader, én invokasjon (~2 t). Samme frosne rekkefølge
  begrenset til vinduet først; ingen epoke fullføres.
- **Observasjon:** TRAIN-probe 4 096 ved 0/256/1 024/8 050; CONTROL4096
  (uendret) ved 0 og 8 050. Samme seed, tap, batch, ferske vekter.
- **Regel:** samme seks parvise familiekorrigerte grenser ved 20 d mot initial
  og mot konstanten for de trente vindusradene; 365-dagersfasens resultat
  rapporteres ved siden av, velges aldri. Hypotesen: signalet holder eller
  styrkes med to år nær-regime-data; svekkelse leses som recency-forfall.
  Full epoke krever fortsatt kvalifisert signal og eget vedtak.

## Hva planen ikke åpner

Ingen featurekassering, ingen ny arkitektur, ingen separat Exit-modell, ingen
TEST-lesing, ingen full epoke eller full VAL, ingen seleksjon på CONTROL, ingen
hyperparametersøk, ingen live/paper/broker/ordre/spending. GC-order-flow er
parkert. Sammenligningen mot D1-ridge-journalen på samme beslutningsdager
(steg 6 i NATIVE_D1_CONTEXT_HORIZON_DECISION_20261011.md) kommer etter P3 og
krever egen binding.
