# Begrenset Entry-innlesing ved kontrolloppstart — 2026-09-17

Første CPU-oppstart ble drept med exit137 før INITIAL_STATE eller RESULT.
Logg, plan, script og minne-cgroupens OOM-hendelse er bevart.20G/512M-grensene
beholdes. Ingen modell-forward eller optimizersteg ble kjørt.

Kontrollfabrikkens from_artifacts leste alle101 Entry-kolonner, inkludert
nestede sekvenser, selv om den bare bruker tid. Full TRAIN har313399 rader
og27 891 775 594 ukomprimerte Parquet-bytes. Modellen får features fra
andre, allerede bundne eiere. Én read_parquet får derfor columns=["time"].
Hele kildefilen hashkontrolleres fortsatt. Indeks-, klokke-, kilde- og
prisbindinger, alle features/labels/tidsrammer og modell/trening er uendret.

Fire målrettede tester består. Artefakttesten inkluderer nå nestede96×238-
sekvenser og krever uendret tid uten disse kolonnene, med hele filens hash
bevart. De andre dekker kausale tilstander/priser, gammel VAL-populasjon og
TRAIN-ID-er over den gamle5508-grensen. Bare from_artifacts-AST er endret.

Én faktisk retry med samme inputs og hardwaregrenser gjenstår før minne-
blokkeringen er løst i praksis. Ingen refit eller fullført måling gjentas.
Bevis: handover_snapshot/PREFIX_ENTRY_CLOCK_READ_FIX_20260917.json.
