# Veien videre — vurder den fullførte læringsprøven

## Nøyaktig stoppunkt for neste agent

Native hovedencoder-prøve er fullført på256 steg. Final ONLINE er lagret,
checkpointfilen er rehashet, guard PASS, task Disabled, ingen native prosess.
Brukt scope er stengt. Ingen paret læringsanalyse eller nytt verdict er laget.
Ny initialmåling og tidligere representasjonsdiagnose skal ikke gjentas.

1. Kjør handover --check fra riktig inngang. Les CURRENT_HANDOVER.md,
   NEXT_RUN_POLICY.json og docs/MAIN_ENCODER_FIXED256_20260918.md.
   current_work viser siste kjøring; eldre toppnivåfelt er historikk.
2. Gjenbruk REVIEW_OPERATOR.py fra den fullførte residual256-mappen. Tilpass
   artefaktbindinger til hovedencoder-prøven og NY native initialmåling.
   Ikke kjør gamle PREPARE_REVIEW/REVIEW/RECORD-operatorer uendret: de binder
   eldre kilde, runtime, initialbaseline og verdict. Analysecommit og trenings-
   commit er nå ulike. Bevar originalscriptene; skriv ny operator i ny run-mappe.
3. Utfør én CPU-analyse med gx1_capped_run.sh --class audit. Bruk bare lagrede
   TRAIN-prediksjoner og eksisterende koordinater. Verifiser receipt/cursor,
   tensor-/lærerhash, optimizer256, eksakte targets/masks/cohort og nye initial-
   bindinger. Native målparitet og checkpointfilhash er allerede kontrollert.
4. Rapporter Entry LONG/SHORT, Exit-ankre og samplede Exit-states for alle ni
   TRAIN-måneder: MSE, sentrert feil, korrelasjon, bias og spredning, fellesverdi
   og LONG−SHORT-kontrast, handlinger samt valgt referanseverdi/regret. Sammenlign
   ny initial, residual256, kausal256 og TRAIN-konstanter, inkludert FLAT=0.
   Gamle ONLINE-outputs gjenbrukes; ikke kjør historiske vekter gjennom ny modell.
5. Skriv PAIRED_TRAIN_REVIEW.json og VERDICT.json i gjeldende run-mappe. Vurder
   både Entry og Exit; lavere bias, større variasjon, all-FLAT/all-HOLD eller
   Entry alene er ikke PASS. Oppdater denne overleveringen og commit/push.

Dette er gjenbrukt TRAIN med fitted-overlapp. Referanseverdi er ikke realisert
profitt. Ved klar læring må en separat kronologisk vurdering bindes før kjøring;
TRAIN-fit, senere kvalitet og samlet kostnadsjustert økonomi er separate porter.
Ved svakt resultat: én konkret årsak i lagrede bevis, minste nødvendige rettelse.
Ingen automatisk forlengelse, ny initialisering, brede søk eller gjentatte tester.
TEST forseglet. Ingen ny native kjøring er autorisert av denne overleveringen.
