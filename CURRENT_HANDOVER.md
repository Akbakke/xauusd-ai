# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, gren `work/gx1-current`.
`GX1_RULES.md` er bindende. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus;
prosess og eksakte terminalkvitteringer overstyrer prosa. Tidligere bølger finnes i Git.

## Gjeldende status

`PAUSED_M5_COLD_FIT_CONSUMED_EDGE_UNPROVEN`. Ingen GX1-jobb kjører. Den ene kalde
M5-fit-planen er fullført og konsumert. `training_enabled`,
`native_training_enabled`, `full_epoch_training_allowed` og
`native_launch_authorized` er `false`. Full GC, TEST, ONLINE, broker og
handel er stengt. Ingen økonomisk edge eller uavhengig generalisering er bevist.

## Pågående fullrevisjon 17.11.2026

Brukeren ba om hele-prosjekt-sanity check, avgrensede agenter og sletting av
bevist foreldede filer. 2 069 sporede filer / 984 774 tekstlinjer er maskinelt
skannet; alle 1 353 Python-, 625 JSON- og 9 shellfiler parser. Dette er ikke
manuell semantisk godkjenning av hver linje. Tre agenter reviderer henholdsvis
feature/modell, mål/tap/score og kjøring/retention. Ingen trening eller TEST.

Konkrete retention-funn: eieren ignorerte `PermissionError` i `/proc/*/fd` og
`fdinfo` og kunne derfor feilaktig godkjenne sletting uten fullt
åpen-skriver-bevis. Minste fail-closed rettelse og to negative tester er gjort;
162 fokuserte tester besto. Fysisk normal bruker får nektet 27/31 fd-tabeller;
root kan lese alle. Root-capped audit med 4G/512M er verifisert og en ytre
`flock` på andre2s eksisterende prosjektlås blokkerer parallell jobb på
samme inode. Ingen DATA/RUNS er slettet. Retention-eieren har laget én plan
for `HISTORY2009W_EARLY_DECISION_20260927` (2 600 617 478 bytes): plan SHA
`3e3150233e10c4e19413f0e2f8f82cf4a0d951a8c5e2a9ba412c2a81df7a1bc7`.
Godkjenning/utføring er ikke kjørt.

## Gjort og målt 17.11.2026

- Regelen om direkte M5-edge, aktiv mismatch-kontroll, sparsom tokenbruk og
  ingen arbeid for å fylle ventetid står i `GX1_RULES.md`. Avgrensede agenter
  reviderte kilde → input → fasit → tap → prognose → valg → kostet utfall,
  featurekonsum og opprydding. Målte kald-rute-feil ble rettet før fit:
  senere dekning, Fold1-prior, uobserverte enheters profil og SCORE-scope.
- Eksakt kaldt view fikk fysisk Oct–Dec-dekning på 17 489 rader. Rå FIT-/SCORE-
  gather, fersk normalisert modellinitial og kosttap besto før én samlet plan.
  Prefiksnormaliseringen fra 29 574 tidligere FIT-rader ble gjenbrukt; 275
  gaprader ble ekskludert. Den gamle 4 096-radersdekningen tilhører et annet
  view og ble ikke brukt. Ingen inputvakt ble omgått.
- `TA_M5_COLD_JOINT_FIT_20261117_001`, plan SHA
  `f6d2d368ebc48280dd688a35c58182a63401384a01a90a5eb0722f04037697b6`,
  kjørte én gang på fysisk boot 664 under kapslet trainer-vakt. Kildefilenes
  innhold var uendret gjennom kjøringen. 29 574 FIT-rader, 1 849/1 849
  optimizersteg, 1 849 treningsforwards og 2 × 1 094 senere scoreforwards;
  null hoppede batcher. Kjøretid 6 395,58 sekunder. Terminal `COMPLETE`, exit 0.
  RESULT SHA `716fc82d6e2e13499dd2dce38b0274c00d4ab11e7d303952069b5d5c07d5e9e7`;
  TERMINAL SHA `b9c8fff72e3213d074c8498091750384503f19568268206b9fbc26f9835760bc`.
  Checkpoint SHA `deaf8d615198c7203e71ea9a486cb8187f12cdde33d29207034365a82f071c14`;
  SCORE-capture SHA `72eb98b91642dc4b73f713d57952f566ce6c495e3f0beff3250763eea7ab3f7f`.
  Terminalreferanser, antall og checkpoint-state-hash er kontrollert uavhengig.
- Fersk initial valgte 17 489 FLAT. Trenet modell valgte 8 289 LONG,
  0 SHORT og 9 200 FLAT. 7 365 valgte terminaler er kjente: kostet netto
  −6,622 bps i snitt; 924 er ukjente. Av 8 289 valg har 6 112 komplett
  prisbane og 2 177 ukjent bane. På komplett valgte baner var predikert
  kvalitet (netto minus pre-MFE MAE) +3,159 bps mot faktisk −22,173 bps.
  Predikert netto var +17,990 mot faktisk −5,593 bps; predikert MAE
  14,831 mot faktisk 16,580 bps. Kvalitetsgapet er −25,332 bps.
- På 11 318 komplette 480-minuttersbaner rangerte modellen LONG best
  i 97,06 %, fasiten i 53,45 %. Side-treff var 52,78 % og korrelasjon
  mellom predikert og faktisk sideforskjell 0,035. Korrelasjon for LONG-
  kvalitet var 0,043. Kvalitetssnitt for valgte komplette baner var
  −2,045 bps i oktober, −22,787 i november og −26,203 i desember.
  Forhåndsvalgt alltid-LONG på alle komplette h480-baner ga −8,216 bps
  netto markout; dette er ingen posisjonsført handelsreferanse.
- Den eldre varme utviklingsmodellen valgte 0 LONG og hadde +3,246 bps
  predikert mot −16,167 faktisk kvalitet på 4 031 komplette valgte baner.
  Den kalde modellen endte med ensidig motsatt sidevalg. Forskjellig
  FIT-periode og view gjør at årsaken ikke kan tilskrives tapsendringen alene;
  den skapte ikke dokumentert side- eller prisbaneedge. Gjenbrukt historisk
  utvikling kan ikke bevise uavhengig edge.
- Oppryddingsrevisjonen fant ingen repo-fil eller DATA/RUNS-artefakt som
  var bevist trygg å slette etter referanse-, import-, test- og eierskapskontroll.
  Ingen sletting ble gjort. Målt ledig plass var 543 GB. DATA/RUNS eies av
  retention-regelen i `GX1_RULES.md`. Gammel handover-prosa er fjernet her;
  historikken ligger i Git.

## Ikke gjort og neste direkte steg

Ingen ytterligere fit, TEST-lesning, forward-journal, ONLINE-baseline, broker eller
handel er åpnet. Posisjonsført PnL, alle åpne posisjoner og uavhengig
økonomisk generalisering er ikke målt. Ikke bruk positive terminalsnitt,
FLAT eller tekniske PASS som edge.

Ikke start flere modellrunder fra dette resultatet. Neste nyttige kontroll
må være én lesende, fast sammenlikning av om de allerede konsumerte TA-/TF-
inputene faktisk bærer stabil kostjustert side- og kvalitetsinformasjon i
både tidligere FIT og senere utvikling. Sammenlikn med forhåndsvalgt
alltid-LONG for lange horisonter. Et nytt inngrep krever et konkret målt
avvik og en samlet plan; TEST forblir forseglet. M5-forward-journal krever
separat vedtak.
