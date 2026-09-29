# Veien videre — oppdatert 29.09.2026

Gjeldende status eies av [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md). Bruk bare
`/home/andre2/src/GX1_CURRENT`, `work/gx1-current`. Én agent og én tung jobb innen CURRENT.

## Fullført autorisert forberedelse

V37-inputbygging, fersk post-rebuild/readiness, TRAIN/VAL lifecycle-filbindinger,
repo-gjennomgang/testtriage og den bestilte kompleksitetsvurderingen er fullført.
Sluttrapport:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/FINAL_PREPARATION_REVIEW_20260929/FINAL_PREPARATION_REPORT.json`.
Les eksakte bevis via NEXT_RUN_POLICY.json. Alle videreføringer er konsumert.
Bevar opprinnelig RED og ny eksplisitt readiness-recovery; ingen ny rebuild.

- [Kompleksitet og redundans](docs/FEATURE_COMPLEXITY_REVIEW_20260928.md).
- [Repo-dekning, rettelser og teststatus](docs/REPO_REVIEW_20260928.md).
- [Byggets feilrettinger og historikk](docs/NATIVE_PREPARATION_AND_REPO_REVIEW_20260927.md).

Ingen ny fullsuite eller gjentatt bestått datakontroll uten konkret nytt funn.
Alle features, åtte familier og tidsrammer beholdes. Ingen data/checkpoints slettes;
ekstern diskopprydding krever retention-eierens rekkeviddebevis og godkjente plan.

## Neste arbeid: godkjent A/B/C-forskning

Følg [forskningsplanen](docs/TA_RESEARCH_PLAN_20260929.md). Dokumentasjonsbølgen
er publisert som 52c8761e. Ridge/konstant-rapportering og sterkere regularisering
er nå mekanisk kontrollert. Ingen ny markedsmåling er gjort. Neste steg er:

1. Reparer sammenhengende kjøp-og-hold/finansiering, risikosammenligning og
   inferens/styrke hos eksisterende instrumenteiere.
2. Commit kjørbare forhåndsregistreringer for sju D1-felt (A), navngitt makrotillegg
   (B) og fem frosne fortsettelsesceller med utførelsesdiagnostikk (C).
3. Kjør avgrenset gjennom capped audit/producer. Gjenbruk gyldige cacher og
   sammenlign på samme kausale populasjon. Ingen gamle planer relanseres.
4. Dokumenter GO/NO-GO/INKONKLUSIV og foreslå neste operatørbeslutning.

Planen er ikke en kjørbar forhåndsregistrering. Ingen nye A/B/C-resultater finnes
ennå. Regel 1 tillater kun navngitte, manifestbundne eksterne forskningsinputs.
training_enabled=false: native optimizer/trening, full native VAL, TEST-utfall,
live/paper og spending er stengt. Bare den registrerte forskningens dataadgang
og CPU-fits er åpnet. Native mål-/horisontkontrakt krever senere eget vedtak.

## Fullførte historiske spor

- [Konsolidering og tidligere sletting](docs/CONSOLIDATION_20260926.md).
- [Tidlig kalibrering og historisk beslutningskontroll: NO-GO](docs/HISTORY2009W_EARLY_DECISION_RESULT_20260927.md).
- [Modellfrie baselines](docs/MODEL_FREE_BASELINES_RESULT_20260927.md),
  [makrohendelser](docs/MACRO_EVENT_BASELINES_RESULT_20260927.md) og
  [intradag-mekanismer](docs/INTRADAY_MECHANISMS_RESULT_20260927.md).

Disse planene skal ikke relanseres. Målet om kostnadsjustert edge er ikke oppnådd.
