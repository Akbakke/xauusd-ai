# Gjeldende GX1-mål — 2026-09-13

Tren GX1 mot positiv kostnadsjustert netto Bps med hele det avtalte feature-settet og samarbeid mellom timeframes og familier. Ett års TRAIN og full juni-VAL er fullført. Hovedløpet bruker hele femårsgrunnlaget, opptil 30 epocher, juni-VAL etter hver og early stopping med patience 5. TEST er forseglet; ingen live-/papirhandel eller ekstern spending.

Første femårs-TRAIN er fullført og bevart: 313 399 rader, 19 588 optimizersteg, checkpoint 309. Brukeren har autorisert gjenstart for bedre utnyttelse. Exit-VAL økes fra batch 16 til 128; TRAIN og Entry-VAL er fortsatt 16. Modell, target, optimizer, EMA, scheduler, RNG, rekkefølge og modellvalg videreføres. Juni-VAL starter med nye akkumulatorer.

Siste klargjorte kilde er f40ec16f. Første batchforsøk avdekket manglende cuDNN FP32-innstilling; begge TF32-flagg er nå av. Den neste faktiske sammenligningen viste identiske 256 hold/exit-valg, Q-avvik 0,000049 Bps og 1,5873 ganger raskere modellberegning, men en for streng diagnostisk toleranse stoppet den. Rettelsen bruker eksplisitt absolutt grense 0,0001 Bps og identiske valg. Tre målrettede kontroller og Git-hooks består. Total VAL-fart og fremdrift på den nye kilden må bekreftes; tidligere vekter påstås ikke å være trent under den korrigerte cuDNN-innstillingen.

Neste handling: fullfør den autoriserte oppstarten fra bevart TRAIN, bekreft faktisk VAL-fremdrift og mål tilstander per sekund. Ikke gjenta trening, smoker eller analyser. Følg deretter den eksisterende kampanjen omtrent hvert 15. minutt, stille ved normal fremgang. Én agent og én tung jobb.

Ressurser: 300 W fysisk grense, 85 °C kjerne, 80 °C minne og 12 GiB VRAM; keeper senker til 200 W ved 80 °C kjerne. Alle data/features, full juni, kostnader og lært Exit beholdes. Positiv samlet Bps og liveklarhet er ennå ikke dokumentert.

CURRENT_NATIVE_RUN.json har eksakte kilde-, runtime-, recipe- og planbindinger. Ta over via CURRENT_HANDOVER.md og SYSTEM_MAP.md. Checkpointkopien på Mac er kontrollert; Git er ikke full rådatabackup. Daterte eldre observasjoner er historikk. Bevar fryste kilder og checkpoint-/VAL-historikker.
