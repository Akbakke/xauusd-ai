# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende; NEXT_RUN_POLICY.json eier arbeidsstatus.

## Aktivt mål og neste handling

TA_DIRECTION_EDGE_CAUSE_CURRENT_COMPLEMENT_ADMITTED. Edge-målet er fortsatt uoppnådd.
Én ny, avgrenset CPU-evaluering er admittert i market_state_ta_current_complement_20261112_001.
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_CURRENT_COMPLEMENT_20261112_001.
Plan: configs/research/MARKET_STATE_TA_CURRENT_COMPLEMENT_20261112_001.json.
Kjør én gang via eksisterende capped producer12GiB/512MiB. Ingen optimizer eller fitting.
Claim/result/terminal er autoritative; planen konsumeres også ved feil. Ingen automatisk relaunch.
Root eier policy, launch og samlet kontroll; kilde og policy fryses under kjøringen.

Spørsmål: skader de siste4096-radenes oppdateringer også andre opprinnelige TRAIN-rader,
eller finnes forverringen bare i senere periode? INITIAL bef35daf og begge faktiske sluttstater
fra MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001 evalueres på sammeCPU1/no_grad.
CURRENT INITIAL finnes allerede i ANCHOR_SPAN_SHORT_LEARNING_20261112_002; ingen migrering.

Fast utvalg:46jevnspredte hele komplementbatcher,736rader, pluss to separate bookendbatcher
med32rader pertilstand. Én modell, trevekts-/bufferlaster,144forwards, maks600modellsek/900wallsek.
De siste6radene i opprinnelig TRAIN er utenfor dette avgrensede B16-utvalget.
Alle genuine inputs og alle femTF bevares. Gjenbruk anchor.prepare_fold1 og sampled-S/pass0,
native-F evaluate_fit_batch. Ingen nye features, normalisering, targets eller senere modellskåring.
Primært M5S/F side-Brier/log mot eksaktINITIAL og frossenprior; alleTF beholdes.
ParetUTC-dag/uke2000bootstrap, ingen gruppe-/modell-/horisontutvelging.

Dette er opprinnelig TRAIN, ikke urørt holdout. Det er et736-raders utvalg, ikke fullkomplementet.
De32kontrollradene holdes utenfor skårene; CPU/GPU-paritet der er ingen global feilgrense.
Det gamle E3-komplementresultatet gjaldt annen modell og blandet numerikk; gammel plan forblir konsumert.

## Kvalifisering og foregående resultat

SOURCE_CHECK.json: syntetiske fortegn/AUC/masker og bookend-separasjon bestått for alle10endepunkter.
THREAD_CHECK.json: faktisk intra/inter1 etter standardinitialiseringen. Loaderens eksisterende tellere
initialiseres og kreves3. Uavhengig read-only kildekontroll fant ingen gjenværende konkret blokkering.
SOURCE_REVIEW.json binder kvitteringene; docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json samler bevisene.

Foregående native-horisontkontroll er COMPLETE/CONSUMED. FULL TRAIN-AUC M15–D1 var0,62–0,71,
senere0,47–0,51, med svakere sannsynligheter ennprior. Samme låste støtte-/EMA-/HTF-oppsett
viste ingen avklart merverdi mot alltid-LONG på noen av femhorisonter. AUC er ikke winrate.
Tid, nativeATR-skala og gyldigstøtte endres sammen; ingen isolert ventetidseffekt er bevist.
Resultatet finnes uendret i MARKET_STATE_TA_SETUP_NATIVE_HORIZONS_20261112_001/RESULT.json.

Global training/native_training, full epoch og fullVAL er false. Ingen tung jobb er startet ennå.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion er stengt; GC er på pause.
Korte runder er kvalifisering, ikke en generell dom over prosjektets kapasitet.
Full kvalifisert TRAIN-historikk og30–40epoker vurderes senere; ingen bred trening admittert nå.
