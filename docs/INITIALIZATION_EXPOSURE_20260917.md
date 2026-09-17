# Startvektene har allerede sett TRAIN — 2026-09-17

**Ingen undersøkt initialisering i den aktive kildekjeden kan gjøre en senere
bit av eksisterende TRAIN til ukjente data.** Dette er en avgrenset, målt
konklusjon om bundne forløpere, ikke en påstand om alle filer på disken.

- Eldre V9 checkpoint152:77 312 unike rader i fullført loader-prefiks, fordelt
  over alle60 måneder fra juni2021 til mai2026. Siste brukte Entry er
  2026-05-29T14:00Z.16 016 rader er fra siste treningsår.
- Det bundne native seed har deretter fullført65 295 ettårs-Entries;
  epochkvittering, progresjon og signert rekkefølge viser alle én gang.
- Femårsforgjengeren315 har fullført én hel epoch over313 399 rader og320
  batcher i neste. Dette er før overgangen til dagens ettårskjede.
- Inputnormaliseringens bundne bevis omfatter313 399 Entry-rader og1 764 423
  Exit-beslutninger. VAL- og TEST-fit er0. Den er lovlig TRAIN-normalisering,
  men ikke normalisering fittet bare før en ny retrospektiv TRAIN-kontroll.

Loader-eksponering er ikke en påstand om at alle oppgaver hadde gyldig fasit
på hver eneste rad. Månedsfordelingen og eksakt rekkefølgehash er i JSON.
Gamle seed-PASS og fullføringskvitteringer er historisk bevis, ikke kjøreordre.

## Følge for generalisering

Å velge andre Entry-ID-er, flytte et datofilter etter trening eller nullstille
bare Exit/Entry-hodene fjerner ikke eksponeringen i modellens øvrige vekter.
Den tidligere påviste targetoverlappingen er dermed bare én av grunnene til
at de interne TRAIN-kontrollene ikke beviser generell læring.

Juni er fortsatt negativt, gjenbrukt utviklings-VAL. Disse funnene viser ikke
framtidslekkasje i native juni-evaluering eller at kausale features mangler
signal. De viser at retroaktivt holdout i TRAIN krever et rent startgrunnlag.

## Neste ene arbeidspakke

Forbered én forhåndsbundet kronologisk læringskontrakt. Behold samme arkitektur,
alle200 features, åtte familier, MTF, økonomi og native vakter. Et kontrollert
ueksponert startpunkt eller fersk initialisering av samme modell må brukes;
normalisering må fittes bare på treningsprefikset. Bind kalendergrense og
observerte fasitvinduer før utfallsanalyse. Fasitvinduer som krysser grensen
må ikke skjules eller behandles som en handelsregel om maksimal holdetid.

Kontrakten må også binde online-/target-/EMA-/optimizerstart, konsistente
Entry/Exit-mål, budsjett/sluttpunkt, baselines, retninger og senere tidsperioder.
En kald Exit-lærer som gir bare umiddelbar kostnad er ikke i seg selv korrekt
Entry-supervisjon; den allerede dokumenterte målkjedesvikten må håndteres i
forsøkets begrunnelse, ikke bare flyttes til nye vekter. Ingen brede vektsøk.
Tidligere undersøkte perioder skal fortsatt kalles gjenbrukt utviklingsdata;
ferske vekter gjør dem ikke til en forskningsmessig urørt sluttest.

Den konkrete native blokkeringen er målt i
`gx1/scripts/run_unified_exit_random_access_full_train_v1.py:371`:
modellen konstrueres, men laster så bundet ferdigtrent EMA og bevarer gammel
inputnormalisering. Beskriv minste kompatible endring i disse eksisterende
eierne før kode endres. Ingen ny arkitektur, separat runner eller omgåelse av
bindings-/sikkerhetskontroller. Planleggingssteget åpner ingen forward, fit,
targetberegning, optimizersteg, native-kjøring, full epoch/VAL eller TEST.

## Kontroll og originalbevis

CPU-vakt4GiB/512MiB swap, én tråd;14,69s. Checkpointet ble lest med mmap for
metadata og epoch_order. Ingen modell ble konstruert, ingen modell-/optimizer-
tensorer ble beregnet med, og ingen kode, checkpoint eller treningsdata endret.
State-, pointer-, datasett-, manifest-, normaliserings- og receiptbindinger
brukt som dateringsbevis er hashkontrollert. Konklusjonen gjenbruker tidligere
kontrollerte overgangsbevis og innfører ingen ny trening.

Kilde:e25125fdcb755975e0c51f777fb5b4b90e01924d.
Original RESULT og OPERATOR.py:
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_REFERENCE_POLICY_20260916_REFERENCE/FROZEN_READOUT_GENERALIZATION_20260916/INITIALIZATION_EXPOSURE_20260917

RESULT SHA256:871674a608c39eff48b0f66b6391727bbb68c466ea1c5a27e523505c6ec65968

Speil:handover_snapshot/INITIALIZATION_EXPOSURE_20260917.json.
