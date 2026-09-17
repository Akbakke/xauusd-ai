# Veien videre — GX1, 17. september 2026

## Første handling for neste agent

Kjør `./handover.sh --check` fra Mac-overleveringen. Les CURRENT_HANDOVER.md,
RUNNING_NATIVE_CALIBRATION.json, NEXT_RUN_POLICY.json og læringsporten.
Ikke anta at et gammelt prosess-/checkpointnotat er nåstatus. Arbeid som én
agent med én tung jobb. Alle fullførte checkpoints, mål og resultater skal bevares.

Fjernkontakt: `ssh gx1-3090-lan`; Linux via
`wsl.exe -d Ubuntu-22.04 -u andre2 -- /bin/bash -s`.
Alle kildeoperasjoner skjer i `/home/andre2/src/GX1_CURRENT`.

## Prediksjonsavviket er målt — ingen aktiv jobb

Fire forwards er fullført på e53645d7, samme cachede TRAIN16 og frosne initial/
finalmodeller. Native sluttkvittering: 18:34:13 UTC / 20:34:13 Oslo, guard PASS,
trainer/observer 0. Faktisk boot459, forberedt runtime-navn BOOT458. Windows-task
deaktivert, controller avsluttet, null optimizersteg og originale checkpoints bevart.

Inferens matcher begge lagrede prediksjoner eksakt. Gradientmodus avviker med
maks 0,0002992153 Bps initialt og 0,0001640320 Bps til slutt; ingen handlingsbytter.
Den gamle 0,0001 Bps-kontrollen mellom ulike modus består altså ikke. Toleransen
er uendret. Dette forklarer signaldiagnosens måleblokkering, ikke læringssvikten.
Det er ikke isolert hvilken enkeltkernel som gir forskjellen.

Den testede rettelsen gjelder bare diagnostikken: verifiser vanlig inferens mot lagret
inferens med samme toleranse, og beregn/rapporter gradientmodus separat. Bevar
handlingskontroll og synlig numerisk avvik; ikke påstå eksakt samsvar mellom modus.
Deretter kan en særskilt bundet signaldiagnose finne hvor variasjon/gradientsignal
går tapt. Rettelsen er implementert og fem målrettede tester består.
En ny signalplan må bindes før kjøring; ingen ny trening er åpnet.
Den brukte parity-planen er stengt og skal ikke relanseres.

Bevis: `handover_snapshot/ENTRY_FORWARD_PARITY_{RESULT,REVIEW}_20260917.json` og
`BASE/NATIVE_ENTRY_FORWARD_PARITY_20260917/REVIEW.json`. Her er BASE den vanlige
prebuilt-roten i GX1_DATA. Entry/Exit-læringsporten er fortsatt ikke bestått.

## Referanser for sammenligningen

Felles rot (`BASE`):
`/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912`.

- Korrekt baseline: `BASE/CAUSAL_ENTRY_TRAIN_BASELINE_20260917/RESULT.json`,
  SHA256 `53d85f84c2d5ae6e3b53ce88c8e9f0422a889a670c24ad39f388ed214833cf86`.
- Dens `DERIVED_TRAIN_BASELINE.json`, SHA256
  `cf816df7ab079bd1605480bf0bbb4c07b6a5901a7426a2d4ad5260287eb363fc`,
  inneholder korrekt Entry-target og uendrede initialprediksjoner. Exit-mål
  leses fra den bundne originale native TRAIN-observasjonen.
- Initial: `BASE/NATIVE_PREFIX_INITIAL_MEASUREMENT_20260917/INITIAL_MEASUREMENT_AUDIT.json`.
  Bruk bare TRAIN-observasjonen; ikke les CONTROL på nytt.
- Forrige connected256: `BASE/NATIVE_ENTRY_CONNECTED_FIXED256_20260917/PAIRED_TRAIN_REVIEW.json`
  og `.gx1-candidate-training-session.CANDIDATE_BUNDLE/final_online_measurement/TRAIN_OBSERVATION.json`.
  Entry-prediksjonene må vurderes mot den nye avledede fasiten; gamle Entry-MSE
  er ikke direkte sammenlignbare. Exit-målene er uendret.
- Fullført treningsplan, ikke relanser: `BASE/NATIVE_CAUSAL_ENTRY_FIXED256_20260917/PLAN.json`, SHA256
  `0f48d7aec4a530ec5df44df60d76332377ee81f72584a11ca235818d699cc63b`.
- Fullført treningsrecipe: samme mappe `/NATIVE_RECIPE.json`, SHA256
  `8b96b88a28ddf513062a7608cfdc3fa08286bc093d6eeb4fa12279418f944bcd`.
- Campaign-plan: samme mappe `/CAMPAIGN/CAMPAIGN_PLAN.json`, filhash
  `45d038f9198ff682bbf9e437f511d6aa92d55e563ba8fa2a837a1f403c5dc016`.
  Ikke forveksle filhash med planens semantiske hash.

## Kriterier for senere arbeid — siste prøve er allerede avvist

Rapporter Entry LONG/SHORT MSE, sentrert feil, korrelasjon, prediksjons-/target-
fordeling, fellesverdi og LONG−SHORT-kontrast. Ta med LONG/SHORT/FLAT-valg og
referanseverdi/regret mot TRAIN-konstanter, inklusive FLAT=0. Rapportér alle
ni måneder. Bedre bias alene eller all-FLAT er ikke tilstandsavhengig kvalitet.

Rapporter Exit på både anker og alle samplede states, begge sider: feil,
konstanter, fordeling, HOLD/EXIT-valg og referanseverdi. Entry alene oppfyller
ikke samlet mål. Skill referanse-policyverdi fra faktisk lærte handelsutfall.

Hvis læring og beslutninger forbedres tydelig: bind neste kronologiske
kontroll før utførelse etter eksisterende læringsport. Mars–mai og juni er
allerede utviklingsdata; de skal ikke omtales som urørt holdout. TEST er forseglet.
Samlet økonomi må inkludere kostnader og åpne posisjoner. Ingen automatisk
full epoch/full VAL, live/paper eller spending.

Hvis resultatet uteblir: lokaliser én konkret resterende årsak i de eksisterende
mål-/gradient-/outputbevisene. Ikke ny epoch på håp, tapsvektssøk, flere features,
refaktorering eller justering av terskler på samme kontrollutvalg.

## Operatører og publisering

Native256, signalfeilen og parity-målingen er avsluttet. Alle tilhørende
operatører er nå historiske og skal ikke relanseres. Bevar cache og ferdige
resultater. En ny signaldiagnose må få egen plan etter den minimale målerettelsen.
Ingen vekter, rådata eller hemmeligheter skal pushes.

Stående godkjenning gjelder offentlig kode, dokumentasjon, interne filstier
og aggregerte resultater til `Akbakke/xauusd-ai`, `work/gx1-current`. Commit/push
ferdig arbeid når native-kilden ikke lenger er frosset; ikke spør samme spørsmål.
Målet er fortsatt uoppfylt og skal ikke markeres komplett på teknisk PASS.
