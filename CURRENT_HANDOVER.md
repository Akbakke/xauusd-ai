# GX1 — overlevering 17. september 2026

## Siste forsøk fullført og avvist

Residualnormaliseringsprøven på88310075 fullførte256 steg og sluttmåling med
guardPASS og eksakt bevart fasit/lærer. Paret analyse er ferdig: Entry fortsatt
FLAT256/256; Exit fortsatt HOLD for LONG og EXIT for SHORT. Små MSE-endringer
er blandet, og begge Entry-sider taper mot TRAIN-konstanter. Læringsporten
er ikke bestått. Ingen jobb kjører, task er Disabled og brukt scope er stengt.

Kode: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Data: /home/andre2/GX1_DATA. Mac er overleveringskopi. Start med
./handover.sh --check på Mac eller bash scripts/gx1_handover.sh --check i Linux.
current_work er nåstatus; COMPLETED_RUN og gamle VAL-felt er historikk.

Les docs/RESIDUAL_NORMALIZATION_REVIEW_20260917.md og VEIEN_VIDERE.md.
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Siste artefaktmappe er BASE/NATIVE_RESIDUAL_NORMALIZED_FIXED256_20260917 med
PAIRED_TRAIN_REVIEW,VERDICT,RESIDUAL_BOUND_AUDIT og COMPLETION_REVIEWED.json.
Checkpoint5,epoch0/offset256,256 steg er bevart. complete=false/RESUMABLE
gir ingen resume-tillatelse. Ingen modellkjøring er åpen i policyen.

## Hva vi vet og neste avgrensning

Før normalisering viste TRAIN16 ti ganger mindre Entry-hidden-variasjon og
stor nesten felles skalavekst. Den nye kandidaten begrenser de tre endrede
residualkorreksjonene analytisk, men gir samme svake beslutninger. Dette
beviser at denne endringen alene ikke løser problemet. Det viser ikke hvor
gjenværende variasjon forsvinner. Rå MTF og hoved-fuse er ikke dekket av
de tre normgrensene. Ikke legg til flere arkitekturendringer på antakelser.

Neste: kontroller først om bevart TRAIN16-cache kan kobles eksakt til nye
native TRAIN-prediksjoner og sluttcheckpoint. Deretter kan én eksisterende
native representasjonsmåling bindes for å skille rå local/MTF/context fra
Entry-hidden. Ingen ny optimizer, ny trening eller brede søk er åpnet.
Ikke bruk gamle ikke-null-checkpoints med ny arkitektur og kall dem gamle
baselines; gjenbruk deres originale outputs. Kandidatkoden er bevart for
reproduserbar diagnose, ikke godkjent som bedre modell.

Entry-detach og etterpåklok target-klipping er tidligere rettet. Entry-mål
er første likvidasjonsverdi+(119/120)*Q_HOLD, med negative verdier beholdt og
ugyldig/terminal HOLD=0. Bevar Exit-mål,kostnader,bootstrap og kausalitet.
120 beregningssteg er ingen maksimal handelsholdetid.

Én agent/én tung jobb. Alle200 features/åtte familier/tidsrammer bevares.
TEST er forseglet; mars–mai/juni er utviklingsdata. Ingen full epoch/full VAL,
CONTROL/TEST, live/paper eller spending. Ingen fast taps-/holdetidsgrense.
Stående offentlig push gjelder kode,docs,interne stier og aggregater,
aldri rådata,vekter eller hemmeligheter. Gjenbruk ferdige kontroller.
