# Veien videre — oppdatert 29.09.2026

Gjeldende status eies av [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md). Bruk bare
`/home/andre2/src/GX1_CURRENT`, `work/gx1-current`. Én agent og én tung jobb innen CURRENT;
all test-/datakjøring bruker eksisterende capped-run-vakter.

## Gjeldende rekkefølge

1. SMC-rettelsen er kontrollert: 189 fokuserte og 92 integrasjonstester;
   full lokal M1-kontroll og faktisk MTF-paritet på de sju berørte TRAIN-radene.
   Ny prosjektlås består 294 tester og faktisk parallell låsadgang. Bevisene
   gjenbrukes; ingen ny fullsuite eller gjentatt diagnose uten et nytt funn.
2. Fullfør `CONTINUE_LOCKED_QUOTES_20260929` under
   `HISTORY2009W_NATIVE_V37_20260928`, med fersk outputrot
   `DATASET_LOCKED_QUOTES_RECOVERY_20260929`. Forrige TRAIN-skriver fullførte
   652 552 rader uten OOM; lifecycle stoppet på BID lik ASK. Samme prisregel
   som kanonisk tape er nå kontrollert med 44 tester og faktisk TRAIN-vindu.
   Ingen priser er endret. Datasettet mangler fortsatt ferdigmanifest.
   Les samme runs START/TERMINAL og prosesser; ikke dupliser planen.
   Ferdige upstream-inputs gjenbrukes bare etter hash-/kodekontroll og ny preflight.
   Bevar kildefrys, delvise gamle datasett og originale checkpoints.
3. Fullfør native-inputs, post-rebuild/readiness og lifecycle-bindinger innen
   [det autoriserte forberedelsesomfanget](docs/NATIVE_PREPARATION_AND_REPO_REVIEW_20260927.md).
   Teknisk PASS er ikke en tillatelse til å starte trening.
4. Avslutt gjenstående kompleksitetsvurdering: parameter-/beregningsfordeling,
   redundans utover de 67 kandidatene og begrunnelse for hjelpeoppgavene.
   [Vurderingen](docs/FEATURE_COMPLEXITY_REVIEW_20260928.md) foreslår færre
   hjelpeoppgaver som én mulig senere sammenligning; ingen blind featurefjerning.

Den tidligere fullsuiten og triagen er dokumentert i [repo-gjennomgangen](docs/REPO_REVIEW_20260928.md);
alle de 18 daværende feilede tilfellene besto avgrenset ny kontroll. Fullsuiten
skal ikke gjentas uten en konkret ny grunn.

## Opprydding og avslutning

Rydd motstridende gjeldende status, frakoblet kode og dokumentert overflødige
filer. Bevar unike resultater, data-/runtime-avhengigheter og checkpoints.
Sletting under GX1_DATA må gjennom retention-eieren med rekkeviddebevis og
hashbundet plan/godkjenning/kvittering. Katalogalder er ikke slettingsgrunnlag.
Avslutt med fokuserte kontroller, `git diff --check`, oppdatert overlevering og
commit/push av ferdig arbeid etter stående autorisasjon.

## Fullførte historiske spor

- [Konsolidering og tidligere sletting](docs/CONSOLIDATION_20260926.md).
- [Tidlig kalibrering og historisk beslutningskontroll: NO-GO](docs/HISTORY2009W_EARLY_DECISION_RESULT_20260927.md).
- [Modellfrie baselines](docs/MODEL_FREE_BASELINES_RESULT_20260927.md),
  [makrohendelser](docs/MACRO_EVENT_BASELINES_RESULT_20260927.md) og
  [intradag-mekanismer](docs/INTRADAY_MECHANISMS_RESULT_20260927.md).

Disse planene skal ikke relanseres. De gamle operative instruksene finnes i
Git-historikken; bare gjeldende omfang ovenfor er en videreføringsinstruks.
`training_enabled=false`: ingen optimizer, native trening, full VAL, TEST-
utfall, live/paper eller spending. En fremtidig trening krever egen bundet policy.
