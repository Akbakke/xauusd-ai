# Veien videre — oppdatert 29.09.2026

Gjeldende status eies av [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md). Bruk bare
`/home/andre2/src/GX1_CURRENT`, `work/gx1-current`. Én agent og én tung jobb innen CURRENT.

## Oppfølging 30.09.2026: undersøk B-kildene

Første kildeprøve er fullført og bevart: arkivmetadata bekrefter to historiske
kopier, men CDX og ALFRED-POST fikk timeout. Begge COT-datoforespørslene
returnerte samme kopi 07.04.2019, etter gullrevisjonen. Oppfølgingen bindes i
configs/research/TA_B_SOURCE_SNAPSHOT_PROBE_20260930.json: bare disse to
arkivkopiene og samme lille ALFRED-prøve med skjemaets manglende submit-felt
rettet. Den rettelsen er ikke bevis for årsaken til timeouten.

Brukeren ba «Ja undersøk B». Dette åpner en avgrenset kildeundersøkelse:
GLD/COT-versjoner, arkivmetadata og ALFRED-transport. Fullført A/C og tidligere
B-resultat bevares. Ingen ny fit, native trening, TEST, handel eller spending.

Før nye kildebytes hentes bindes
configs/research/TA_B_SOURCE_REOPEN_PROBE_20260930.json: fem forespørsler om
arkivmetadata og én liten DFII10-prøve for juni 2025. En byte-lik kopi av
CURRENTs rene HTTP-hjelpere kjøres som transport på Mac fordi samme skjema
tidligere svarte der og fikk timeout på WSL. Rå svar og hasher føres tilbake til
CURRENTs runtime; Mac er ikke en forskningskodebase. Ingen arkivsnapshot eller
modellinput godkjennes av denne transportprøven.


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

## Gjeldende grense: A/B/C-planen er avsluttet

[Samlet beslutning](docs/TA_RESEARCH_DECISION_20260929.md):
A/C INKONKLUSIV, B ikke målt grunnet kildebegrensning. Resultater, terminaler og
etterkontroller er ferdige. Ingen GO og ingen ny kjøring autorisert.
Bevar [A](docs/TA_A_RESULT_20260929.md), [B](docs/TA_B_RESULT_20260929.md)
og [C](docs/TA_C_RESULT_20260929.md); ingen omkjøring eller parameterletning.

Neste mulige operatørbeslutning er om dokumenterbare historiske GLD/COT-versjoner
kan kvalifiseres før full B eventuelt gjenåpnes. Datatilgjengelighet og
felles A/B-populasjon må bindes før en ny fit. Ingen kostnad eller leverandørvalg
er vedtatt. C ga ikke den nødvendige GO til utførelses-/ordrebokforskning;
A ga ikke grunnlag for ny native mål-/horisontkontrakt.

training_enabled=false. Native trening/full native VAL, TEST-utfall, handel og
spending er stengt. Alle native familier, checkpoints og tidligere artefakter
bevares. Målet om lønnsom bot er fortsatt ikke oppnådd.

## Fullførte historiske spor

- [Konsolidering og tidligere sletting](docs/CONSOLIDATION_20260926.md).
- [Tidlig kalibrering og historisk beslutningskontroll: NO-GO](docs/HISTORY2009W_EARLY_DECISION_RESULT_20260927.md).
- [Modellfrie baselines](docs/MODEL_FREE_BASELINES_RESULT_20260927.md),
  [makrohendelser](docs/MACRO_EVENT_BASELINES_RESULT_20260927.md) og
  [intradag-mekanismer](docs/INTRADAY_MECHANISMS_RESULT_20260927.md).

Disse planene skal ikke relanseres. Målet om kostnadsjustert edge er ikke oppnådd.
