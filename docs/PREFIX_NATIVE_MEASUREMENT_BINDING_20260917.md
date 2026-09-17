# Fryste utvalg koblet til native måling — 2026-09-17

TRAIN256-ID-er og alle fire sampletilstander per Entry er gjenbrukt eksakt fra
det fryste4096-utvalget. CONTROL256-ID-ene er uendret. Kontrollens fire states
er nå beregnet én gang med eksisterende native full-populasjon epoch0-strøm,
inkludert opprinnelig chunk-indeks. Valget bruker bare identiteter, counts og
klokker. Ingen priser, fasitverdier, fit eller modell ble brukt i forberedelsen.

Begge roller har256 Entries og1024 samplede Exit-tilstander. TRAIN dekker
alle ni opprinnelige måneder; kontrollen har83/87/86 Entries i mars/april/mai.
All TRAIN-støtte slutter senest2026-02-25T12:14Z før mars1; kontrollens støtte
går senest til2026-06-01T00Z, nøyaktig tillatt grense.17 TRAIN- og13 kontroll-
samples har sensurert observasjonsslutt; bootstrap beholdes. Ingen samples
er byttet, fjernet eller valgt ut fra resultater. Forberedelsen tok27,4sek CPU.

Eksisterende kohorteier binder begge rollene til disse uforanderlige filene.
TRAIN-probens opprinnelige, usorterte native rekkefølge beholdes. Den faktiske
fysiske indeksfilen, parent-/child-ID-er, tider, counts og cutoff kontrolleres.
Målerapporten henter samplevalgene fra den fryste bindingen og avviser andre
sampler eller kildeindeks før modell-forward. Målekohorter kan ikke brukes
til økonomisk rollout; senere økonomivurdering krever eksisterende separate
autoritet etter læringsporten.

Native komponentoppstart oppretter en grunn kopi av TRAIN-datasettet før
livsløpsadapteren kobles. Den deler uendrede inputs og riktige TRAIN-labels,
men utløser ikke treningssampleren under før-/ettermåling. Ingen ekstra
innlesing av features, ny modell, normalisering eller optimizer er innført.

18 målrettede syntetiske kontroller består på første forsøk, inkludert
eksisterende komponent-/rapportintegrasjoner. Begge faktiske256-kohorter
er deretter bygget og kontrollert med de samme native eierne på CPU.
Rå koordinater ligger bare i GX1_DATA; repository-resultatet er aggregert.

Ingen faktiske modellprediksjoner, targets eller optimizersteg er beregnet.
Neste er å koble og binde den første target-/prediksjonsmålingen fra lagret
INITIAL_STATE.pt gjennom eksisterende native campaign og vakter. Deretter
gjenstår det ene256-stegsforsøket og fast ONLINE-sluttmåling. Ikke gjenta
utvalgsberegningen, initialiseringen eller ferdige tester. Ingen læring er bevist.

Bevis: handover_snapshot/PREFIX_NATIVE_MEASUREMENT_BINDING_20260917.json.
