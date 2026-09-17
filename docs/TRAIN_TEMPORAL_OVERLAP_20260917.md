# TRAIN-kontrollene deler markedsforløp — 2026-09-17

**Den separate TRAIN128-kontrollen er ikke uavhengig generaliseringsbevis.**
Ulike Entry-ID-er fjernet direkte dubletter, men ikke delte prisbevegelser.
Ingen modell-/treningskode er endret. Ingen ny forward, fit, targetberegning,
optimizersteg, VAL eller TEST er utført. Den frosne kandidaten forblir avvist.

## Målt overlapp

- Fitted TRAIN:512 Entries/2048 tilstander. Separat TRAIN:128/512.
- Direkte Entry-overlapp:0. Begge utvalg er tidsmessig sammenflettet fra
  juni2025 til mai2026, ikke senere kronologiske kontrollperioder.
- 331/512 separate tilstander (64,65%) deler minst én observert reward-overgang
  med fitted TRAIN; dette berører118/128 Entries (92,19%).711 kryssende par.
- 62 separate fasitvinduer er helt dekket av treningsvinduene.24 538 av60 569
  successor-observasjoner deles (40,51%, teller gjentatte kontrollobservasjoner).
- 32/67 state0-kontroller har targetoverlapp. Alle512 kontrolltilstander deler
  minst én bar i sin480-bars lokale inputhistorikk med fitted TRAIN.
- Observerte targetlengder er3–120 TRAIN og9–120 kontrolloverganger. Eksakte
  dataklokker brukes, også over markedspauser; ingen antatt120min veggklokke.
  Seneste targetgrense er2026-06-01T00:00Z, slutten av siste TRAIN-M1-bar.

## Lagrede resultater uten nytt fit

Konstanten nedenfor er beregnet på hele fitted TRAIN i den eksisterende
målingen. Oppdeling etter overlapp er etterfølgende beskrivelse, ikke et nytt
valg av treningsdata, terskel, holdout eller beslutningsport.

| Separate tilstander | Antall | LONG kandidat / TRAIN-konstant MSE | SHORT kandidat / TRAIN-konstant MSE |
|---|---:|---:|---:|
| Alle |512|1006,74 /1071,23|1052,83 /1070,11|
| Med targetoverlapp |331|1129,69 /1203,25|1162,41 /1203,95|
| Uten targetoverlapp |181|781,89 /829,81|852,42 /825,35|
| State0 med targetoverlapp |32|324,38 /396,16|340,39 /398,44|
| State0 uten targetoverlapp |35|637,08 /623,90|731,74 /620,42|

Den samlede SHORT-forbedringen holder ikke i gruppen uten targetoverlapp.
Ved state0 taper begge sider mot konstanten uten targetoverlapp. Dette er
forenlig med for optimistisk intern kontroll, men gruppene har ulike utfall
og er små: forskjellen isolerer ikke kausal effekt av overlapp.

JSON-feltet baseline_mse bruker original95 fra den frosne targetpakken.
Det er ikke base96-verdien fra VALUE_READOUT_PROBE som ble brukt i forrige
semantikkrapport. Konstant- og kandidatverdiene ovenfor er direkte
sammenlignbare; baselineetikettene må ikke blandes.

## Hvordan koordinatene er kontrollert

40 inputcacher og deres frosne mål/baseliner er hashkontrollert. Noen gamle
batchrapporter mangler states. Derfor avleses heltallsalderen entydig fra
cachefeltet bars_in_trade med eksisterende FP32-normalisering: bare én fysisk
heltallsverdi i det mulige TRAIN-området gir akkurat samme lagrede verdi.
Dette er eksakt oppslag, ikke avrunding av en omtrentlig invers. Begge sider,
pathlengde og alle576 tilgjengelige direkte state_index/decision_time-poster
samsvarer. Startkoordinater kommer fra den bundne TRAIN-indeksen og M1-klokken.
Alle boundary-HOLD-actions er gyldige, slik at ingen tidligere absorberende
terminal forkorter en sides observerte rewardstøtte.

Fasitoverlapp måles som identiske M1-overganger (start,end], bekreftet både
med intervall/prefikstelling og eksplisitt mengdeunion for hver kontroll.
Lærer-bootstrap kan gi mer avhengighet enn dette målet dekker. Felles kausal
inputhistorikk er ikke i seg selv framtidslekkasje. Ingen nye faktiske
modellinputs eller targets materialiseres. Målingen tok28,66s under CPU-vakt
4GiB/512MiB swap, én numerisk tråd. To tidligere leseforsøk stoppet på gamle
JSON-formatforskjeller før RESULT; operatorene er bevart ved siden av V3.

## Beslutning og neste konkrete steg

Ingen videre tilpasning til disse kontrollene, deres delgrupper eller juni.
Neste ene lesekontroll skal spore dokumenterte treningsdatoer for eksisterende
initialiseringer/checkpoints. En senere kronologisk kontroll må også være
usett av initialiseringsvektene; hele den observerte fasitstøtten må skilles over
treningsgrensen. Purging er en evalueringsgrense, ikke maksimal holdetid.

TRAIN-tilpasning må fortsatt bevises, men disse kontrollene kan ikke alene
åpne generaliseringsporten. Juni-resultatet forblir separat negativt bevis.
Ingen antakelse om at ny arkitektur, mer trening eller endret tapsvekt løser det.

## Originalbevis

Kilde:0a421e9cc418ac1a868481d9ffb194e9ba4a316b.
Original RESULT og OPERATOR.py ligger i:
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_REFERENCE_POLICY_20260916_REFERENCE/FROZEN_READOUT_GENERALIZATION_20260916/TRAIN_TEMPORAL_OVERLAP_20260917_V3

RESULT SHA256:9c67415c09f52fb2f550da31e4d4547c75fc7d2692b5a83f6956bc69ef53a304

Speil:handover_snapshot/TRAIN_TEMPORAL_OVERLAP_20260917.json.
