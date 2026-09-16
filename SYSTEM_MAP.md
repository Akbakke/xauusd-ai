# Gjeldende GX1-systemkart — 2026-09-16

Rå M1-priser/bid/ask og native feature-eiere
→ TRAIN-eid normalisering og bundne M1/M5/MTF-data
→ Entry: M5 + M15/H1/H4/D1, LONG/SHORT/FLAT
→ Exit: M1 + M5/M15/H1/H4/D1, HOLD/EXIT_NOW
→ samme V4-økonomi-/kostnadseier i trening og native evaluering.

Alle200 features, åtte familier og tidsrammer beholdes. Featureverdier tilhører
sine lukkede klokker. Fremtidige prisutfall brukes som læringsmål, aldri som
online-input. Entry-forecast har et selvstendig signal og undersøkt isolasjon
på fire rutingsparametere; backbone er delt og handelsverdiene bruker fortsatt
Exit-læreren. Det er ikke en fullt uavhengig Entry. Exit får lokal prissti,
livstidssammendrag med MFE/MAE og MTF-kontekst. Sammenkobling beviser ikke nytte.

Opt-in femstegs Exit-backup følger den frosne lærerpolicyen og observerte M1-
successors, med bootstrap etter beregningsgrensen. Det er ingen fast holdetid.
Bare frosne target-forwards deles i mindre delbatcher for å holde minnegrensen;
online-trening, sampler og én backward beholdes. Læringsgevinst er ikke bevist.

Én kjørevei: NEXT_RUN_POLICY.json → kilde-/databundet native campaign → laststyrt
GPU-clock-launcher/Windows-controller → gx1_capped_run.sh → native kandidatvindu.
TRAIN16, VAL256, åtte CPU-arbeidere, tre timers VAL-vinduer, FP32/TF32 av og
etablerte maskinvarevakter. Full epoch/VAL er blokkert mens læring er uavklart.

Handover er kun lesing: current_work beskriver nåstatus; COMPLETED_RUN.json er
bundet historisk bevis. Ingen gamle smoker eller separate VAL-kjørere er
alternative oppstartsveier. Importer med v12/live i navnet kan fortsatt eie
nødvendige offlinefunksjoner; navn alene er ikke grunnlag for sletting.

Neste beslutning følger docs/LEARNING_GATE_20260916.md. TEST forblir forseglet.
