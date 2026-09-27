# Modellfrie grunnlinjer — resultat 27.09.2026: NO-GO (scalp og swing)

Forhåndsregistrert i [MODEL_FREE_BASELINES_PREREG_20260927.md](MODEL_FREE_BASELINES_PREREG_20260927.md)
(commit aca19cae, før kjøring). Kjørt på ren kilde aca19cae; tape
`XAU_M5_NATIVE_2009_20260701_PAIR_20260927` lest til 2025-05-30 20:55 (1 140 232 M5-barer);
beslutninger 2011-06-01 → 2025-05-31; bundet kostpolicy a48f8e56…. Evidens:
`/home/andre2/GX1_RUNS/MODEL_FREE_BASELINES_20260927/results.json` (sha256 av innholdet
c0d8b699…). Evidensklasse: målt på ekte deklarerte bytes. VAL og TEST er ikke lest.

**Ingen av 62 celler er GO eller LOVENDE.**

## Scalp (M5, mot FLAT) — 26 celler

| regel | handler | brutto bps | treff | netto bps | t |
|---|---|---|---|---|---|
| MOM k∈{1,6,12}, h∈{6,12,19} | 47–162 k | −2,40 … −2,56 | 0,37–0,43 | −6,41 … −6,59 | −56 … −187 |
| REV (samme) | 47–162 k | −2,30 … −2,55 | 0,39–0,45 | −6,35 … −6,60 | −55 … −183 |
| ORB London | 3 605 | +0,56 | 0,488 | −3,66 | −3,3 |
| ORB New York | 3 528 | −0,84 | 0,456 | −5,01 | −5,4 |
| Sesjon Asia long / short | 3 661 | −1,70 / −5,85 | 0,47 / 0,43 | −6,23 / −9,85 | −9 / −14 |
| Sesjon London long / short | 3 652 | −3,63 / −1,43 | 0,47 / 0,47 | −7,99 / −5,43 | −10 / −7 |
| Sesjon NY long / short | 3 660 | −3,48 / −2,11 | 0,47 / 0,47 | −7,91 / −6,11 | −7 / −6 |

- Momentum og reversering har begge brutto ≈ −2,4 bps: det er spread-kryssingen alene; ingen
  retning på 5 min–95 min med disse reglene. Kost per rundtur ≈ 6,4 bps (spread ≈ 2,4 + 2 × 2 bps).
- Største brutto-effekt: Asia-timene stiger i snitt ~+2 bps mid per sesjon (long −1,70 mot short
  −5,85 brutto), London-åpningsbrudd +0,56 bps — begge langt under kostnaden, også uten slippage.

## Swing (D1-klokke, mot alltid-LONG) — 36 celler

Alltid-LONG netto per beslutning (scenario A): −9,9 bps (1 uke), −7,8 (2 uker), −3,0 (4 uker);
forventet ≈ −4 per uke fra drift − finansiering − kost, innenfor SE ≈ 7,7 bps.

Beste celler (Δ mot alltid-LONG, scenario A):

| celle | Δ bps | t | Δ scen. B | bjørnefold Δ | positive år |
|---|---|---|---|---|---|
| tsmom_126_lf_h5 | +9,1 | 1,87 | +5,1 | +24,6 | 38 % |
| sma200_lf_h5 | +7,6 | 1,53 | +3,3 | +21,9 | 54 % |
| combo_lf_h5 | +6,9 | 1,30 | +1,4 | +22,9 | 54 % |
| tsmom_252_lf_h5 | +5,3 | 1,20 | +1,0 | +21,3 | 54 % |

- Long/flat-trendfiltre slår alltid-LONG bare ved å stå utenfor i bjørnemarkedet 2011–15
  (+20–40 bps per beslutning der) og taper det meste tilbake i oksemarkedene; ingen når t ≥ 2 eller
  60 % positive år.
- Long/short-variantene er dårligere: shortene taper mer i oksemarkedene enn de tjener i
  bjørnemarkedet (tsmom_63_ls_h10 Δ −31,9; tsmom_21_ls_h20 −60,8).
- Scenario B (uten finansiering) gjør alltid-LONG sterkere og krymper alle Δ.

## Hva dette betyr

- **Scalp:** enkle retningsregler har null brutto retning etter spread, og kostnaden (~6 bps per
  rundtur med policyens slippage, ~2,4 bps uten) er større enn noen målt brutto-effekt. Et
  scalp-system må finne > 2,4 bps brutto per handel før slippage bare for å gå i null; verken disse
  reglene eller de tidligere modellmålingene (ridge/HGB på alle felt, 23.–26.09) har vist noe i
  nærheten.
- **Swing:** trendfølging i gull gir risikoreduksjon i fallende marked, ikke signifikant
  meravkastning over 2011–2025 etter kost. Retningsgevinst utover drift er ikke påvist med enkle
  regler på XAU-data alene.
- Etter registreringens beslutningsregel: modellarbeid må begrunnes med noe nytt, ikke flere
  features.

## Ikke undersøkt

- Realistisk kost per megler (spread + slippage er policyens antakelse; brutto-tallene over er uten
  slippage og viser at scalp også da er negativt).
- Sommertid i faste UTC-sesjoner.
- Andre modellfrie regler (volatilitetsbrudd, nivå-/range-regler på H4/D1) — ikke registrert her.
- VAL (2025-06 → 2026-06) er urørt og kan brukes til å bekrefte et fremtidig kandidatfunn.
