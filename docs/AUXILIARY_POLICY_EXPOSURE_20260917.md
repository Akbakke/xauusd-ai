# Eksponerte hjelpefasiter — 2026-09-17

Før en ny kronologisk læringsmåling må fasitreglene avgrenses til før
2026-03-01. Nåværende retning-/path-policy og størrelses-ECDF er begge fittet
fra2021-06-01 til2026-05-31T23:50Z. Sistnevnte brukte266740 av313455 kandidater.
De er dermed eksponert for den foreslåtte mars–mai-kontrollen, selv om disse
månedene tidligere het TRAIN og metadata sier at VAL/TEST ikke ble brukt.

| Aktiv loss | Eksponert avhengighet |
| --- | --- |
| Posisjonsstørrelse | Valgt side, tradability-mask og TRAIN-ECDF |
| Sidevis MAE | Fasit ved19 M5-perioder, valgt av full-TRAIN-policy |
| Trendlinjeutfall | Valgt19-bars horisont; trap-fasit bruker også fitted path-terskler, side-margin og retning |
| Dip/forecast/timing/tail/vol |37 rå fasiter med faste horisonter; må purges på virkelig M5-tidslinje opptil96 bars |
| Entry/Exit | Rettet felles referanse; samplede forløp og bootstrap må ligge før cutoff |

37 rå fasiter bruker den bevarte feature-tidslinjen før ufullstendige mål
fjernes.96 forskjøvne Entry-ID-er er derfor ikke en gyldig klokke. Trendlinjens
hold-label følger full M5-registry, mens path/side/size følger eksakt M1-fill
og M1-utfall. Disse tre tidsstøttene skal ikke erstattes med én antatt varighet.

Eksisterende direction-policy-fit tar train_start/train_end og krever at
alle maksimale kandidathorisonter slutter innen TRAIN. Størrelsesfit binder
samme policy. Gjenbruk disse eierne med den frosne tidligere perioden; beregn
deretter bare avhengige fasiter/masker og bruk frosne policyer på kontrollen.
Ingen ny søkemetode, resultatstyrt horisont, tapsvekt eller modellarkitektur.

Bundet registry og squeeze oppgir fit-slutt2025-05-31, før kontrollgrensen.
Dette er sjekket i metadata; hele parameterberegningen er ikke kjørt på nytt.
Rangeringen er full-TRAIN-eksponert og merket som upstream prerequisite, ikke
runtime-authority. Den skal ikke brukes til nye featurevalg.200 features,
åtte familier og alle tidsrammer beholdes; ingen generell kausalitetsfriskmelding.

Ingen pris-/fasitverdier ble lest i denne auditen, og ingen faktisk fit,
modell-forward, optimizer, GPU eller TEST ble brukt. Kalender-ID-er er ennå
ikke fit-klare. Frosset DESIGN og CONTROL256 er uendret; mars–mai forblir
gjenbrukt utvikling selv etter rettelse. Funnet må ikke omtales som påvist
årsak til tidligere juni-svikt eller dokumentert generell læring.

Maskinbevis:handover_snapshot/AUXILIARY_POLICY_EXPOSURE_20260917.json.
Original RESULT.json og OPERATOR.py ligger i GX1_DATA/.../
LIFECYCLE_V2_FULL_TRAIN_20260912/AUXILIARY_POLICY_EXPOSURE_20260917.
