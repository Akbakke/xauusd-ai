# GX1-status

Trening er stoppet. Separat Exit-klipping er prøvd i én native32-kandidat og
forkastet. Modellen forbedret fit på faktisk trent512 litt, men ble dårligere
på separat TRAIN med samme femstegsmål. Sidevalg kollapset til LONG=HOLD og
SHORT=EXIT; Entry forblir FLAT. Standardkoden er tilbakeført, alle bevis bevart.
Overførbar beslutningskvalitet og lønnsomhet er ikke dokumentert.
training_enabled=false; ingen full epoch/full VAL eller TEST.

CURRENT_HANDOVER.md er eneste gjeldende fortelling. Les ferdig resultat og
neste målkomponentdiagnose i docs/EXIT_PRIVATE_CLIP_LEARNING_20260916.md.
Gjenbruk ferdige trace/reward/Q/input/output-cacher; ikke gjenta32-kandidaten
eller512-analysen. Hent faktisk status med ./handover.sh --check på Mac eller
bash scripts/gx1_handover.sh --check i Linux. Historiske planer er ikke startautoritet.
