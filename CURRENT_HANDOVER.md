# GX1 — overlevering 18. september 2026

**Stoppunkt: native256 er fullført; paret læringsvurdering gjenstår.**
NATIVE_MAIN_ENCODER_NORMALIZED_FIXED256_20260918 avsluttet med guard PASS,
eksakt256 optimizersteg og lagret final ONLINE. Windows-task er Disabled;
ingen native prosess eller controller ble observert ved overleveringen.
Brukt scope er stengt. Ikke kjør BIND/PREPARE/ACTIVATE eller resume på nytt.

Kode: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Data: /home/andre2/GX1_DATA. GX1_ENGINE/.git er felles Git-lagring, ikke startvei.
Mac-mappen er overleveringskopi. Kjør ./handover.sh --check på Mac, eller
bash scripts/gx1_handover.sh --check i Linux. current_work er nåstatus;
COMPLETED_RUN.json og gamle VAL-/checkpoint-felt utenfor current_work er historikk.

Treningskilde: 6b44c23d2b685bbfdaaf0bdeb3b162518101fa0d.
Start: 18. september00:15:39 UTC /02:15:39 Oslo.
Slutt: 18. september01:29:46 UTC /03:29:46 Oslo.
Checkpoint5, slot0, epoch0, offset256. complete=false/outcome=RESUMABLE betyr
avgrenset stopp, ikke tillatelse til å fortsette. Checkpointfilens SHA er kontrollert.
Ingen modell-/treningskode er endret etter a1c4b443; nyere handover-commit er separat.

Hypotesen er parameterfri final LayerNorm i hovedencoder. Opprinnelig lærer
uten denne nye normaliseringen er bevart. Ny nullstegsbaseline er fullført og
auditiert; gammel initialprediksjon er ikke riktig baseline for endret ONLINE.
Native sluttmåling bekrefter uendrede targets/koordinater mot ny initial.
Dette beviser ennå ikke bedre læring, generalisering eller lønnsomhet.

**Neste handling:** følg VEIEN_VIDERE.md og docs/MAIN_ENCODER_FIXED256_20260918.md.
Tilpass eksisterende CPU-analyse av lagrede outputs til ny initial og sluttmodell.
Sammenlign Entry OG Exit mot residual256, kausal256 og TRAIN-konstanter, begge
sider og alle TRAIN-måneder. PAIRED_TRAIN_REVIEW.json og VERDICT.json finnes ennå
ikke for denne prøven. Ikke start nye forwards eller trening for å lage analysen.

Bindingsfiler, kvittering, checkpoint-hasher og gjenbruksoperatører står i
handover_snapshot/MAIN_ENCODER_FIXED256_COMPLETION_20260918.json.
Én agent/én tung jobb. Bevar alle200 features/familier/tidsrammer, kausalitet,
kostnader og originale resultater. training_enabled=false; ingen aktivt kjøreunntak.
Ingen CONTROL/VAL/TEST, live/paper, spending eller brede søk. Stående offentlig
push gjelder ferdig kode/docs/stier/aggregater, uten rådata, vekter eller hemmeligheter.
