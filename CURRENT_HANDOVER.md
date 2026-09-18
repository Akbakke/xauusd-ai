# GX1 — overlevering 18. september 2026

Ny startbaseline er fullført og kontrollert. Én separat256-prøve er bundet:
NATIVE_MAIN_ENCODER_NORMALIZED_FIXED256_20260918. Kjøringens faktiske tilstand
må leses fra current_work/prosess/receipt; bundet er ikke det samme som startet.
Ingen ny læring, generalisering eller lønnsomhet er ennå bevist.

Kode: /home/andre2/src/GX1_CURRENT, work/gx1-current.
Data: /home/andre2/GX1_DATA. Mac er overleveringskopi. Start med ./handover.sh
--check på Mac eller bash scripts/gx1_handover.sh --check i Linux.
current_work gjelder nå; COMPLETED_RUN og eldre VAL-felt er historikk.

Rettelsen er parameterfri final LayerNorm på hovedencoder. Prefix-læreren
kopieres uten denne nye normaliseringen. Native nullstegsaudit bekrefter eksakt
bevart kausal Entry-fasit, original Exit-fasit, koordinater, vekter, optimizer,
EMA, scheduler og RNG. Nye ONLINE-startprediksjoner er lagret. Samme vekthash
betyr ikke samme funksjon; gammel initialprediksjon er ikke ny baseline.
Nullstegskilde a1c4b443, guard PASS, 0 optimizersteg. Brukt task Disabled.

Siste residual256 ga Entry FLAT256/256 og Exit fast valg per side. Diagnosen
målte hoved-fuse L2 1,05→117,71 og 9,51 ganger mindre variasjon etter joint-
normalisering enn initialt. Dette begrunner én hypotese, ikke en læringspåstand.
Ingen modell-/treningskode er endret etter den verifiserte nullstegsmålingen.

Neste: følg VEIEN_VIDERE.md og docs/MAIN_ENCODER_FIXED256_20260918.md.
Bare samme4096 TRAIN-Entries/256 steg og final ONLINE TRAIN256/Exit256/1024.
PLAN/operatører ligger under BASE/NATIVE_MAIN_ENCODER_NORMALIZED_FIXED256_20260918;
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Forbered og aktiver én gang. Ikke relanser aktiv eller avsluttet plan.

Én agent/én tung jobb; kilden fryses under kjøring. Bevar alle200 features,
åtte familier/tidsrammer, kausalitet, kostnader og originale checkpoints.
Ingen full epoch/VAL, CONTROL/TEST, live/paper, spending eller brede søk.
training_enabled=false; bare eksakt bundet256-scope er åpnet. Ingen automatisk
utvidelse. Offentlig push av ferdig kode/docs/stier/aggregater er stående godkjent;
rådata, vekter og hemmeligheter er unntatt. Gjenbruk beståtte kontroller.
