# GX1-status — 17. september 2026

Kausal Entry-prøve på kilde955abf19 er fullført: 256 oppdateringer og native
TRAIN-only sluttmåling, guardPASS/trainer0/observer0. Ingen native prosess;
Windows-tasken er deaktivert. Launch-unntaket er stengt. Ingen ny trening nå.

Korrekt Entry-baseline og uendrede Exit-targets er bekreftet eksakt bevart.
**Paret analyse er ferdig: læringsport ikke bestått. Entry FLAT256/256 og
MSE dårligere enn TRAIN-konstanter; Exit alltid HOLD for LONG / EXIT for SHORT.
Alle fire Exit-MSE er dårligere enn forrige kandidat. Neste er årsaksdiagnose
fra eksisterende bevis, ingen ny trening. Generalisering/profitt er ikke bevist.**

Les CURRENT_HANDOVER.md og VEIEN_VIDERE.md. Kjør ./handover.sh --check på Mac
eller bash scripts/gx1_handover.sh --check i Linux for fersk status.

Signaldiagnosen er terminal med prediksjonskontrollfeil, uten signalrapport.
Én separat plan er bundet for å måle avviket: samme TRAIN16, fire forwards,
ingen backward/optimizer eller toleranseendring. Se CURRENT_HANDOVER.md.
