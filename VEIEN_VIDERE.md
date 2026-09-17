# Veien videre — rett hovedbanen med uendret lærergrunnlag

1. Gjenbruk den fullførte input- og representasjonsdiagnosen. Ingen ny forward
   er nødvendig for å bekrefte de målte norm-/variasjonsendringene.
2. Avklar minste bevaring av lærerfunksjonen før modellrettelse. Den native
   treneren bruker deepcopy(model) for target_model; originale vekter alene
   bevarer ikke funksjonen dersom online-arkitekturen endres. Les konstruksjon
   og restore i gx1/models/entry_v10/entry_v10_ctx_train_v3.py, samt prefix-
   initialisering/sluttmåling i gx1/scripts/run_unified_exit_random_access_full_train_v1.py.
3. Velg bare én konkret rettelse av den målte, unnormaliserte hovedbanen.
   Bevar mål/inputs/økonomi og kontroller startmodell/lærer før eventuell
   separat bundet, kort native læringsprøve. Ikke nye port-/tapsvekt-/modellsøk.
4. Krev bedre tilstandsavhengig Entry OG Exit mot sammenlignbare frosne
   baselines. Større representasjonsvariasjon alene er ikke PASS.

Nå: ingen aktiv kjøring eller tillatt ny modellplan. Fullført diagnose finnes
under BASE/NATIVE_RESIDUAL_REPRESENTATION_20260918/REVIEW.json.
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Modellkoden inneholder fortsatt den avviste residualkandidaten; ingen ny
normalisering er lagt til. TEST forseglet. TRAIN-fit, senere kronologisk
kvalitet og samlet kostnadsjustert økonomi er separate porter.
