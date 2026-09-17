# Samme fasit ved samplede Exit-tilstander — 2026-09-17

Den eksisterende bounded-målingen støttet bare state0. Det frosne forsøket
krever også måling senere i handelen; state0 alene kunne skjult Exit-svikt.

Tre eksisterende funksjoner i run_unified_exit_random_access_val_v1.py er
utvidet med eksplisitte samplekoordinater. Referansetrace, priser, gamma og
boundary-forward starter ved riktig offset. Sensurering og bootstrap beholdes;
datogrensen kontrollerer boundary-barens lukking. Ingen ny handelsgrense.

Den eksisterende Entry-rapporten tar nå med alle fire sampletilstander per
Entry og binder koordinatene med hash. Entry-representasjonene gjenbrukes
for både ONLINE og frossen lærer. Entry-fasiten beregnes fortsatt ved state0;
samplefasit kan ikke erstatte den. Hele utvalget kontrolleres før Entry-forward.
Ingen tilstander byttes etter at utfall er sett. Default uten samples er bevart.

26 unike syntetiske tester består: 19 på target-beregningen/eksisterende
state0-kjede, deretter sju nye rapportkontroller og én berørt integrasjonstest.
Ikke gjenta ferdige target-tester. Fire kombinasjoner av offset og levetid
gir bitlik fasit mot native TRAIN, inkludert sensurert boundary. Gjentatte
Entry-ID-er og rekkefølge bevares; feil koordinater og framtidsgrense avvises.
Kun disse tre produksjonsfunksjonene er AST-endret. Modell og optimizer er
uendret. Et quoting-problem i installasjonsskriptet stoppet før kildeendring
eller test; oppsettmetadata ble korrigert uten ny beregning.

Ingen faktisk markedsmodell-forward, optimizersteg eller normaliseringsfit
er utført. Ingen læring eller større kjøreautoritet følger av disse testene.

Neste er eksplisitt TRAIN256-binding og fryste CONTROL-samplekoordinater,
deretter native target-/førmåling fra lagret INITIAL_STATE.pt og dispatch
for fast ONLINE-sluttmåling. Gjenbruk oppstarten; denne målerettelsen endrer
ikke modellvektene. Det ene256-stegsforsøket har ennå ingen læringsresultater.

Bevis: handover_snapshot/SAMPLED_REFERENCE_MEASUREMENT_20260917.json.
