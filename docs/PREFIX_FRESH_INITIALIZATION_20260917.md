# Faktisk fersk native oppstart — 2026-09-17

Eksisterende _build_bound_full_train_components er nå kjørt på faktiske data
med ferdige prefix-artefakter, på CPU under capped producer20G/swap512M.
Ingen historiske modellvekter ble lastet. Modellens 9,617,497
parametere ble opprettet gjennom eksisterende konstruktør med seed20260911.

ONLINE, frossen target og EMA ved null oppdateringer har eksakt samme state-
kontrollsum. AdamW har tom momenttilstand, eksisterende parametergrupper,
lr/weight_decay0.0001 og ingen decay på task-logvar. Task-logvar er nøytrale.
Eksisterende scheduler og EMA-horisont er bevart. Vekter, optimizer, scheduler
og RNG er lagret én gang i INITIAL_STATE.pt under GX1_DATA. Ingen vekter
publiseres; repository-beviset inneholder bare bindinger og aggregerte fakta.

Alle313399 fysiske TRAIN-rader beholdes.47814 er eligible før cutoff; den
forhåndsbundne native rekkefølgen er kontrollert. TRAIN og CONTROL256 bruker
separate label-instanser og samme uendrede råkilde. CONTROL256 har eksakte
forhåndsvalgte IDs og ny prefix-normalisering gjennom native state-fabrikk.

En global forward-vakt avviste enhver modell-forward under konstruksjonen.
Det ble kjørt0 forwards,0 optimizersteg og0 nye normaliseringsfit. Dette er
faktisk komponentoppstart og lagret initialtilstand, ikke læringsbevis.
Retryen tok 621.4 sekunder med målt peak-RSS
9.14 GiB. Første forsøk og OOM-kvittering er
bevart; den ene nødvendige clock-only-rettelsen er nå kontrollert i faktisk
oppstart med uendrede20G/512M-grenser.

Neste er frosne target-/førmålinger gjennom native vakter og dispatch for
fast ONLINE-sluttpunkt. Deretter kan det ene forhåndsdefinerte256-stegs-
forsøket bindes. Ingen større trening, fullVAL, TEST eller handel er åpnet.

Bevis: handover_snapshot/PREFIX_FRESH_INITIALIZATION_20260917.json.
