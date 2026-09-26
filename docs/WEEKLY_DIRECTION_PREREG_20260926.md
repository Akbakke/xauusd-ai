# Forhåndsregistrert ukesmåling av retning — 26.09.2026

Registrert før kjøring. Ingen parameter, horisont, arm eller regel endres etter at
resultatet er sett (GX1_RULES.md regel 2f/22).

## Spørsmål

Forutsier de eksisterende kausale featurene (alle tidsrammer) retningen de neste 1, 2
og 4 handelsukene *utover driften*, fold for fold, 2022-06..2026-05?

Referansen er alltid-LONG på de samme ikke-overlappende blokkene (og konstanten valgt på
fit-perioden), ikke myntkast. En modell som bare er long, får Δ = 0.

## Oppsett (eksakt)

- Instrument: `gx1.scripts.research_entry_direction_walkforward_v1` (schema v2, commit ved kjøring).
- Datasett: `V46_20260825T170935Z_CHAIN/artifacts/V12_FIVE_YEAR_ENTRY_NOTIONAL_20260922T053058Z/dataset`
  (signal v36 = 241). Tape: `GX1_DATA/data/native_xau/XAU_M5_NATIVE_2019_20260804_V4`, avkuttet ved TRAIN-slutt.
- `--early-calibrated-inputs` `GX1_RUNS/V12_EPOCH1_REVIEW_20260923/EARLY_CALIBRATED_FEATURE_INPUTS_20260924/RESULT.json`
  (registry/volatilitet tilpasset til 28.03.2022; 67 lokale blokker + MTF-lanene erstattet).
- `--pattern-primitives-parquet` `GX1_RUNS/.../ENTRY_DIRECTION_WALKFORWARD_20260923/patterns_v1/pattern_primitives.parquet`.
- Armer: `snapshot_mtf` (1 076 felt: M5-signal, ctx, M15/H1/H4/D1-lanene) og `snapshot_mtf_patterns` (1 311).
- Beslutningsklokke: `H4` og `D1` (to kjøringer). M5/M15/H1 er input, ikke beslutningsklokke.
- Horisonter: `1440 2880 5760` tape-barer (≈ 1/2/4 handelsuker); `--targets` bare disse.
- Kost: `--cost-policy` `LIFECYCLE_V2_FULL_TRAIN_20260912/PROSPECTIVE_COST_POLICY_V1/policy.json`
  (sha a48f8e56…): 2 bps per utførelse, provisjon 0, LONG-finansiering 5,4 %/år på veggklokketid, SHORT 0.
- Lærere: ridge (alpha-grid logspace(−2,4,13) + konstant-alternativ), HGB (lr 0,1, min_leaf 20,
  max_iter 100, konstant-alternativ), seed 0, `--inner-fraction 0.2`, `--target-scalings raw`.
- Regler: `argmax_flat` (LONG/SHORT/FLAT etter netto) og `contrast_always_trade` (alltid long eller short).
- Folds: `--fold-boundaries 2022-06-01 2023-06-01 2024-06-01 2025-06-01 2026-06-01` (fire årsholdouts).
- `--statistics nonoverlap`. D1-kjøringen: `--min-fit-rows 150 --min-inner-rows 30` (deklarert, fordi
  ett år har ~240 D1-beslutninger); H4 bruker standardverdiene 1000/100.

## Beslutningsregel (48 celler: 2 klokker × 2 armer × 2 lærere × 3 horisonter × 2 regler)

- **GO:** samlet Δ mot alltid-LONG > 0 med t ≥ 3,1 (ensidig Bonferroni 0,05/48) *og* Δ > 0 i minst
  3 av 4 år.
- **LOVENDE (ikke GO):** t ≥ 2 og Δ > 0 i minst 3 av 4 år — krever bekreftelse på nye data eller
  lengre historikk før noe bygges på det.
- **NO-GO:** ellers.

## Styrke, sagt på forhånd

Ukentlige ikke-overlappende blokker gir ~50 per år og ~200 samlet. Med ukentlig std ~240 bps er
SE for Δ ~17 bps samlet; t ≥ 3,1 krever Δ ≳ 50 bps per uke — større enn driften selv (~35). Målingen
kan dermed bare oppdage en svært sterk betinget retning. Et NO-GO betyr «ikke påvist ved denne
styrken på ett oksemarked», ikke «umulig».
