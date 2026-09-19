# GX1 — overlevering 19. september 2026

Entry fuse256 er fullført og analysert. Begge Entry-sider slår forrige modell
og TRAIN-konstanter på samlet verdi-MSE. Likevel FLAT256/256 og Exit HOLD for
alle LONG / EXIT for alle SHORT. Samlet læringsport er ikke bestått; profitt
eller generalisering er ikke dokumentert. Se docs/ENTRY_FUSE_FIXED256_REVIEW_20260919.md.

Eneste kodevei: /home/andre2/src/GX1_CURRENT, work/gx1-current. Mac er overlevering.
Start med handover --check. Aktuell binding og neste avklaring står i
NEXT_RUN_POLICY.json og VEIEN_VIDERE.md. Ingen native jobb er aktiv eller bundet.

NATIVE_ENTRY_FUSE_NORMALIZED_FIXED256_20260919 avsluttet 13:12:16 UTC /
15:12:16 Europe/Oslo med guard PASS og 256 steg. Treningskilde bdce4ad3;
checkpoint5/slot0/offset256. Windows-task Disabled, brukt scope stengt.
complete=false/RESUMABLE er lagringsstatus, ingen rett til å fortsette.
Ny initialfunksjon, frossen lærer, targets/masker/cohort og tensorhash er kontrollert.

Gjenbruk fullført review, lagrede outputs og beståtte tester. Før mer modellkode:
kontroller eksisterende gradientmålers dekning av felles oppdatering og begge
verdi-hoder. Ingen ny native diagnose er bundet. Ingen nye normaliseringer,
terskel-/tapsvektsøk, full epoch/VAL/CONTROL/TEST, live/paper eller spending.
Målet om bedre Entry/Exit og samlet kostnadsjustert økonomi er fortsatt aktivt.
