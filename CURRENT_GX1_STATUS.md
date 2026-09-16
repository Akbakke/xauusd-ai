# GX1-status

Trening er stoppet. Separat Exit-klipping er prøvd i én native32-kandidat og
forkastet. Modellen forbedret fit på faktisk trent512 litt, men ble dårligere
på separat TRAIN med samme femstegsmål. Sidevalg kollapset til LONG=HOLD og
SHORT=EXIT; Entry forblir FLAT. Standardkoden er tilbakeført, alle bevis bevart.
Overførbar beslutningskvalitet og lønnsomhet er ikke dokumentert.
training_enabled=false; ingen full epoch/full VAL eller TEST.

CURRENT_HANDOVER.md er eneste gjeldende fortelling. Les ferdig resultat og
målt måloppdeling i docs/TARGET_COMPONENT_CAUSE_20260916.md.
Ankerutfall er nå kontrollert på 275/512 Entries; begge sidemiddel er negative.
SHORT har noe rangering, men læringsporten er ikke bestått. Se
docs/ENTRY_ANCHOR_OBSERVED_OUTCOMES_20260916.md. Neste er én begrunnet korreksjon i målkjeden.
Gjenbruk ferdige trace/reward/Q/input/output-cacher; ikke gjenta32-kandidaten
eller512-analysen. Hent faktisk status med ./handover.sh --check på Mac eller
bash scripts/gx1_handover.sh --check i Linux. Historiske planer er ikke startautoritet.
