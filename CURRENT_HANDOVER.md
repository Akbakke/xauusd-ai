# GX1 — overlevering 18. september 2026

Representasjonsdiagnosen er fullført og vurdert. To native inferensforwards,
null optimizersteg, guardPASS og eksakt samsvar med lagrede prediksjoner.
Ingen jobb kjører; Windows-task er Disabled og brukt scope er stengt.

Kode: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Data: /home/andre2/GX1_DATA. Mac er overleveringskopi. Start med
./handover.sh --check på Mac eller bash scripts/gx1_handover.sh --check i Linux.
current_work gjelder nå; COMPLETED_RUN og eldre VAL-felt er historikk.

Konkret nytt funn: hoved-fuse får stor nesten felles amplitude (L2-norm1,05→117,71).
Rå MTF/context varierer fortsatt, men variasjonen etter felles Entry-normalisering
er9,51 ganger mindre enn initialt; Entry-hidden7,07 ganger mindre. Tidligere
residualnormalisering begrenser ikke denne hovedbanen. Læringsporten er fortsatt
ikke bestått: siste trening ga Entry FLAT256/256 og Exit fast valg per side.

Les docs/RESIDUAL_REPRESENTATION_REVIEW_20260918.md og VEIEN_VIDERE.md.
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Siste diagnose: BASE/NATIVE_RESIDUAL_REPRESENTATION_20260918, REVIEW.json og
entry_gradient_diagnostic/RESULT.json. Målekilde bff19fbe; siste treningskilde
88310075. Sluttcheckpoint5/offset256 fra residualprøven og initiallærer bevares.
RESUMABLE/complete=false er ingen tillatelse til videre trening.

Neste konkrete avklaring: native læreren bygges med deepcopy(model). En direkte
normalisering av hovedbanen vil derfor også endre fasitfunksjonen. Finn minste
eksplisitte binding som bevarer original lærer/Entry-/Exit-bootstrap før én
rettelse av hovedbanen. Ingen ny arkitekturendring eller kjøreplan er gjort.

Én agent/én tung jobb. Bevar alle200 features/åtte familier/tidsrammer, kausalitet,
kostnader og native vakter. Ingen blind trening, brede søk, full epoch/VAL,
CONTROL/TEST, live/paper eller spending. Ingen fast taps-/holdetidsgrense.
Offentlig push av ferdig kode/docs/stier/aggregater er stående godkjent;
rådata, modellvekter og hemmeligheter er unntatt. Gjenbruk beståtte kontroller.
