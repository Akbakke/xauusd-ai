# Gjeldende GX1-mål — 2026-09-13

Tren GX1 mot positiv kostnadsjustert netto Bps med hele det avtalte feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

Første femårs-TRAIN er fullført og bevart: 313 399 rader, 19 588 optimizersteg, checkpoint 309. Forrige batch-128-kjøring hadde verifisert automatisk omstart/resume og cirka 265 tilstander/s, men mange posisjoner holder lenge. Et tidligere anslag på 20–30 timer var en upålitelig lineær fremskrivning og er trukket tilbake.

Brukeren avviste tidsbruken. Ny kilde 0e81f5b8 gjenbruker den samme markedsberegningen per absolutt M1-rad innen én frosset VAL-invokasjon. Posisjonens Entry-token, egen historikk, MAE/MFE og hold/exit regnes fortsatt separat. Alle features, juni-data, begge retninger, kostnader og FP32 beholdes. TRAIN-forward og vekter endres ikke. Cachen er tom ved hver ny epoch/invokasjon.

Forrige VAL ble stoppet ved 1 659 288 tilstander; filer og checkpoint er bevart. Ny VAL begynner fra starten med de fullførte TRAIN-vektene. Elleve målrettede kontroller består. Neste handling: fullfør den autoriserte oppstarten, bekreft faktisk GPU-likhet/fart og la kampanjen arbeide. Ingen fullført TRAIN eller full smoke gjentas.

Én agent og én tung jobb. Kontroller omtrent hvert 15. minutt, gjerne sjeldnere ved stabil drift. Endre bare observerte blokkeringer. Stående autorisasjon gjelder. Frosne kilder og fullførte resultater bevares.

Ressurser: 300 W fysisk grense, 85 °C kjerne, 80 °C minne, 12 GiB VRAM. Keeper senker til 200 W ved 80 °C kjerne. Alle kilde-/runtime-/recipe-/planbindinger står i CURRENT_NATIVE_RUN.json. Ta over via CURRENT_HANDOVER.md og SYSTEM_MAP.md. Positiv samlet Bps, nyttig bidrag fra alle ruter og liveklarhet er ikke dokumentert.
