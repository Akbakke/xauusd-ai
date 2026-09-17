# GX1 — overlevering 18. september 2026

## Avgrenset representasjonsdiagnose bundet, ikke startet

CPU-kontrollen er ferdig: cache-inputs, alle16 rader, targets og masker er
eksakt koblet til den nye sluttmålingen. Ny native variant er testet og bundet:
NATIVE_RESIDUAL_REPRESENTATION_20260918, to inferensforwards og null optimizersteg.
Recipe/campaign er ennå ikke forberedt. Følg VEIEN_VIDERE.md og den nye
docs/RESIDUAL_REPRESENTATION_DIAGNOSTIC_20260918.md; gamle planer skal ikke kjøres.

Kode: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Data: /home/andre2/GX1_DATA. Mac er overleveringskopi. Start med
./handover.sh --check på Mac eller bash scripts/gx1_handover.sh --check i Linux.
current_work viser nåstatus; COMPLETED_RUN og gamle VAL-felt er historikk.

Siste treningsforsøk på88310075 er fullført og avvist: Entry FLAT256/256,
Exit HOLD for LONG og EXIT for SHORT. Frosne targets eksakt bevart, teknisk
guardPASS. Læringsporten er ikke bestått. Sluttcheckpoint5/offset256 bevares;
complete=false/RESUMABLE gir ingen rett til å fortsette treningen.
Se docs/RESIDUAL_NORMALIZATION_REVIEW_20260917.md for fullført vurdering.

Måleren kontrollerer originale initialprediksjoner og nye sluttprediksjoner
direkte i inferens. Gamle ikke-null-checkpoints må ikke kjøres gjennom endret
modellkode og omtales som gamle baselines. Gjenbruk originale gamle outputs.
Begrensningen av tre residualkorreksjoner løste ikke beslutningene; ingen ny
modellrettelse er begrunnet før den kommende målingen er tolket.

Én agent/én tung jobb. Alle200 features/åtte familier/tidsrammer bevares.
TEST forseglet; ingen trening/full VAL/CONTROL/handel/spending. Ingen fast
taps-/holdetidsgrense. Stående offentlig push gjelder ferdig kode/docs/stier/
aggregater, aldri rådata, vekter eller hemmeligheter. Gjenbruk ferdige tester.
