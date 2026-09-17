# Native førmåling ved steg null — 2026-09-17

Førmålingen trenger samme native måleeiere og bevart fersk starttilstand som
det fryste forsøket. Den eksisterende budsjettkontrollen avviste nullsteg.
Denne avgrensede utvidelsen tillater ett uttrykkelig bundet nullstegsvindu.
Andre oppsett krever fortsatt positivt steg- eller epochbudsjett.

Lagret INITIAL_STATE.pt gjenbrukes med opprinnelig modell, AdamW, EMA,
scheduler og CPU/Python/NumPy-tilstand. CUDA starter fra samme deklarerte seed.
Eksisterende koordinator lagrer et ordinært initialcheckpoint og stopper før
første TRAIN-batch. Eksisterende måleeier beregner deretter Entry, Exit-state0
og de1024 fryste sampletilstandene for hver av TRAIN256 og CONTROL256.
Begge bruker uendret initialmodell som frossen lærer, med hver sin datogrense.
Målingen bevarer modellmodus, tilfeldig tilstand og aktiv checkpointpeker.
Ingen læreroppdatering, fit, økonomisk rollout eller TEST er åpnet.

12 fokuserte syntetiske CPU-tester består under4GiB-vakt. Direkte trening og
nullstegspause med gjenopptak gir eksakt samme vekter, optimizer, EMA,
scheduler, tilfeldig tilstand og treningsrekkefølge. Tilstand bevares også
ved feil mellom TRAIN- og kontrollmåling. Campaign stopper for gjennomgang
etter ett nullstegsvindu; gamle oppsett kan ikke bruke unntaket.
Faktiske uforanderlige inputbindinger er kontrollert gjennom scope-eieren.

NEXT_RUN_POLICY åpner nå bare dette ene native førmålingsvinduet, med
training_enabled=false og uendrede maskinvarevakter. Kilde skal først være
committet og bundet i recipe/campaign. Faktisk måling er ikke startet.
Dette er teknisk bevis, ingen ny læring eller generalisering.
Rå måleutvalg, observasjoner og vekter forblir i GX1_DATA.
