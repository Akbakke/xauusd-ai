# Gjeldende GX1-mål — 2026-09-14

Tren GX1 mot positiv kostnadsjustert netto Bps med hele feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

BRUKERBESTEMT STOPP: Full juni-VAL etter første femårs-epoch er ferdig. Ingen videre epoch før Entry/Exit, MAE/MFE, månedsslutt uten lukking og mulige Entry-filtre er analysert og forklart. Brukeren har samtidig godkjent lengre kjøreøkter, større VAL-batcher og mer parallell beregning, bare med bevart kvalitet. Disse tiltakene skal måles og verifiseres målrettet; ikke start nytt treningsløp som del av analysen.

Windows-oppgaven GX1RandomAccessCampaignV2 er deaktivert og stoppet; native treningsprosess 723 er avsluttet. Automatikken hadde rukket å starte epoch 2 før stoppbeskjeden kom. Siste lagrede pointer er checkpoint 315, epoch_index 1, 19 908 optimizersteg / batch-offset 320. Første epochs uforanderlige EMA-snapshot (19 588 steg) og alle tidligere kilder er bevart. Kjørbar kilde er fortsatt 03592fe6 i /home/andre2/src/GX1_VAL_PAUSE_ENVELOPE_V40.

Full VAL utførte 57 845 748 tilstandsvurderinger. Av 11 016 hypotetiske LONG/SHORT-forløp ble 7 472 lukket av modellen og 3 544 avkortet ved månedsslutt. Entry valgte 4 180 LONG, 1 328 SHORT og ingen FLAT; bare 2 227 av de 5 508 valgte handlene ble lukket, mens 3 281 ble avkortet. Full-policy netto Bps er derfor ikke autoritativt tilgjengelig. Ikke presenter positiv statistikk bare for lukkede vinnere som hele modellens lønnsomhet.

Sluttresultatet er bevart lokalt i trade_review_20260914/VAL_RESULT_EPOCH_1.json og i den frosne native sesjonen. Analyse skal skille faktiske lærte Exit-resultater fra hypotetisk likvidering ved månedsslutt, og skille Entry-retning/timing fra Exit som slipper tidligere gevinst.

Én agent og én tung jobb. Kontroller én gang i timen, etter brukerens presisering 2026-09-13. Endre bare observerte blokkeringer og uttrykkelig bestilte tiltak. Stående autorisasjon gjelder. Bevar frosne kilder, fullførte resultater og lagret fremdrift.

Ressurser: 20 GiB RAM, 512 MiB swap, 128 oppgaver, CPU 0–18; 300 W fysisk grense, 85 °C kjerne, 80 °C minne, 12 GiB VRAM. Keeper senker til 200 W ved 80 °C kjerne. Eksakte bindinger står i [CURRENT_NATIVE_RUN.json](handover_stage/CURRENT_NATIVE_RUN.json). Positiv samlet Bps, nyttig bidrag fra alle ruter og liveklarhet er ikke dokumentert.

Further verified finding: SHORT HOLD has zero running financing/risk reward. With split-end censoring and no economic terminal, indefinite zero-reward HOLD can dominate voluntarily realizing a loss under the implemented objective. This is an objective-level incentive; more epochs alone are not a demonstrated remedy. Clarify intended economic holding/risk constraints before altering the objective. Full reasoning and evidence are in docs/ENTRY_EXIT_REVIEW_20260914.md. Preserve the stop.
