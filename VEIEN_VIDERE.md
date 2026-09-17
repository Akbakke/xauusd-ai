# Veien videre — GX1, 17. september 2026

## Første handling for neste agent

Kjør `./handover.sh --check` fra Mac-overleveringen. Les CURRENT_HANDOVER.md,
RUNNING_NATIVE_CALIBRATION.json, NEXT_RUN_POLICY.json og læringsporten.
Ikke anta at et gammelt prosess-/checkpointnotat er nåstatus. Arbeid som én
agent med én tung jobb. Alle fullførte checkpoints, mål og resultater skal bevares.

Fjernkontakt: `ssh gx1-3090-lan`; Linux via
`wsl.exe -d Ubuntu-22.04 -u andre2 -- /bin/bash -s`.
Alle kildeoperasjoner skjer i `/home/andre2/src/GX1_CURRENT`.

## Neste handling: én konkret årsaksdiagnose

Native256 og den parete analysen er ferdige. Ikke gjenta dem.
REJECT_EXPANSION_CAUSAL_ENTRY_ALL_FLAT_EXIT_FIXED_BY_SIDE:
Entry FLAT256/256, MSE svakere enn TRAIN-konstanter og dårligere LONG−SHORT-
kontrast enn før. Exit alltid HOLD for LONG / EXIT for SHORT, alle fire
MSE verre enn connected256. Ingen ny trening, CONTROL, VAL eller TEST.

Les docs/CAUSAL_ENTRY_FIXED256_REVIEW_20260917.md og siste artefaktmappes
PAIRED_TRAIN_REVIEW.json/VERDICT.json. Checkpointets payload, modell og frosne
lærer er verifisert; gamle baselinemetrikker er gjenskapt. Originaler er bevart.

1. Følg den tilstandsavhengige LONG−SHORT-feilen til faktisk vektet og klippet
   optimizeroppdatering i eksisterende kilde og lagrede TRAIN-bevis. Bruk
   gx1/models/entry_v10/entry_v10_ctx_train_v3.py og tidligere gradientdiagnose.
   Entry bruker MSE. Ikke anta Huber, manglende features eller ny detach-feil.
2. Gjenbruk lagrede outputs, checkpoint-/optimizerstate og eksisterende
   diagnoseartefakter. Skill liten outputvariasjon fra dokumentert årsak.
   Bevis om dette er svak kondisjonering/oppdatering eller et gjennomsnitt som
   er rimelig under referansepolicyen; ikke press fram handler med terskler.
3. Hvis eksisterende bevis ikke avgjør årsaken, beskriv én nødvendig avgrenset
   måling og bind den særskilt før utførelse. Gjeldende scope åpner ingen nye
   forwards, fit, søk eller trening. Ikke gjør blind ekstra epoch.
4. Rett bare en dokumentert blokkering. Bevar korrigert kausal fasit, features,
   lærer/checkpoints og native vakter. Oppdater handover med konklusjonen og
   push ferdig arbeid under stående autorisasjon.

En eventuell stabil langjobb kontrolleres omtrent hvert 15.–30. minutt eller
sjeldnere. Faktiske prosesser og terminalkvittering avgjør kjøretilstand.

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
- Ny plan: `BASE/NATIVE_CAUSAL_ENTRY_FIXED256_20260917/PLAN.json`, SHA256
  `0f48d7aec4a530ec5df44df60d76332377ee81f72584a11ca235818d699cc63b`.
- Ny native recipe: samme mappe `/NATIVE_RECIPE.json`, SHA256
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

Forberedelse/aktivering og kjøring er fullført. Bevis finnes i siste artefakt-
mappen; aldri kjør dem igjen uendret. Gjenbruk fungerende måleformler og cache.
Lokale arbeidsoperatører finnes under `/private/tmp/gx1-prefix-normalization/`;
varige kopier/bindinger ligger i de respektive artefaktmappene. /tmp er ikke
langsiktig autoritet. Ingen nye vekter, rådata eller hemmeligheter skal pushes.

Stående godkjenning gjelder offentlig kode, dokumentasjon, interne filstier
og aggregerte resultater til `Akbakke/xauusd-ai`, `work/gx1-current`. Commit/push
ferdig arbeid når native-kilden ikke lenger er frosset; ikke spør samme spørsmål.
Målet er fortsatt uoppfylt og skal ikke markeres komplett på teknisk PASS.
