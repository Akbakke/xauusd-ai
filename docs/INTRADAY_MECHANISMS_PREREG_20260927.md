# Intradag-mekanismer (bølge 1) — forhåndsregistrering 27.09.2026

Committet før noe retningsutfall for cellene under er lest. Operatørvedtak 27.09: «kjør bølge 1 nå».
Formål: teste fem konkrete mekanismer for retning på intradaghorisont (1–5 t) i XAUUSD etter kost,
på et utvalg med både stigende og fallende gullmarked. Forskningsarm, aldri Entry-input (regel 1).
Instrument: `gx1/scripts/research_intraday_mechanisms_v1.py`.

Beholdt kodebundet spesifikasjon, ikke ny kjøretillatelse. Avsluttede rapporter
finnes i Git; gjeldende launchomfang eies av NEXT_RUN_POLICY.json.

## Hva som er sett før registreringen

- Sett: modellfrie UTC-sesjoner og ORB (Asia
  ≈ +2 bps mid per sesjon, ORB London +0,56 brutto); makrohendelsene
  (avsluttede rapporter i Git); de 36 oppsettene på V12-radene 2021–26
  (23.–24.09; eneste positive var `pdh_break_trend_H4`, LONG i oksemarked); spread og median |M5-avkastning|
  per UTC-time på 2011–25 og nødvendig treffprosent per holdetid (begge uten retning); 258 ekte
  fyllinger fra mai 2026 (null provisjon, halv spread 0,60 bps i snitt, se under).
- Ikke sett: noe retningsutfall for rundtall, COMEX-klokke-momentum, LBMA-vinduer, sesjoner/ORB på
  lokal klokke, eller oppsettene før 2021.

## Data, fylling og kost

- Tape `GX1_DATA/data/native_xau/XAU_M5_NATIVE_2009_20260701_PAIR_20260927`, lest før 2025-06-01
  (TRAIN-slutt). Inngang ≥ 2011-06-01; historikk fra 2009-06 brukes bare til signaler og oppvarming.
  VAL (2025-06 → 2026-06) og TEST leses ikke; VAL er reservert for bekreftelse.
- Fylling som de modellfrie grunnlinjene: long kjøper `ask_close`, selger `bid_close`; short motsatt.
  Spread er dermed tapens egen bid/ask på hvert tidspunkt.
- Kost: bundet policy `LIFECYCLE_V2_FULL_TRAIN_20260912/PROSPECTIVE_COST_POLICY_V1/policy.json`
  (sha a48f8e56…), finansiering som policyen (scenario A). Slippage per utførelse fra policyens egne
  `val_sensitivity_scenarios`: low 1, central 2, high 4 bps. **Beslutningen tas på low (1 bps per
  utførelse)**; central og high rapporteres. Begrunnelse: de 258 fyllingene policyen bygger på
  (mai 2026) har null provisjon og betalte bare halv spread (0,60 bps i snitt, maks 1,25), som tapen
  allerede dekker; latens-slippage er umålt (ingen klokke fra beslutning til fylling). Low er
  fortsatt 2 bps per rundtur over spreaden.
- Klokker fra tidssonedatabasen (`America/New_York`, `Europe/London`, `Asia/Tokyo`), med sommertid.
  «Pris kl. T» = close av M5-baren som slutter kl. T (starter T − 5 min). Bare hverdager (lokal dato).
  Mangler en påkrevd bar (helligdag, hull), hoppes dagen over og telles.

## Celler (61)

**A. Rundtall (ordreklynger, Osler) — 8 celler.** Gitter G ∈ {10, 50} USD på mid (M5 close/high/low);
en pris på eller over et nivå regnes som over det. Per M5-bar t med forrige close c₋ og close c:
- *Kryss* (følg): et nivå i (c₋, c] → LONG; et nivå i (c, c₋] → SHORT.
- *Avvisning* (fade): intet kryss, og et nivå i (max(c₋, c), high] → SHORT (motstand), eller et nivå i
  (low, min(c₋, c)] → LONG (støtte); begge deler i samme bar = tvetydig, ingen handel.
- Inngang ved close av baren, hold h ∈ {12, 48} barer (1 t, 4 t); ikke-overlappende, grådig kronologisk;
  bare sammenhengende barer t−1 … t+h. Celler `rn{10,50}_{reject_fade,cross_follow}_h{12,48}`.

**B. Oppsett som speilede par — 36 celler.** De 36 regelene i
`research_entry_pattern_setup_edge_v1.declared_setups` (uendret), parvis LONG/SHORT (18 par, f.eks.
`pdh_break_trend_H4` + `pdl_break_trend_H4`, `trend_all_bull_any_bar` + `trend_all_bear_any_bar`).
Primitivene regnes på hele tapen med parameterne fra 23.09-kjøringen (manifest
`GX1_RUNS/V12_EPOCH1_REVIEW_20260923/ENTRY_DIRECTION_WALKFORWARD_20260923/patterns_v1/manifest.json`:
sone 96, FVG 0 ATR, OB 2 ATR/3/5, EQ 0,25 ATR, flagg 5/2 ATR + 3–12/1 ATR, range 24, alderstak 999,
swing 3). Beslutningsrader = alle M5-barer fra 2011-06-01. Side = +1 der bare LONG-regelen slår til,
−1 der bare SHORT-regelen slår til. Inngang ved close, hold h ∈ {12, 48}, samme grådige ikke-overlapp.

**C. Intradag-momentum på COMEX-klokka (New York) — 3 celler.**
- `im_first_to_last`: signal = fortegn(pris 09:20 − pris 08:20); inngang 12:30, utgang 13:30.
- `im_overnight_to_last`: signal = fortegn(pris 09:20 − pris 17:00 forrige hverdag); inngang 12:30,
  utgang 13:30.
- `im_first_hold_to_close`: signal som første; inngang 09:20, utgang 13:30.

**D. LBMA-auksjonen (London 10:30 og 15:00) — 6 celler.** Per auksjon T:
- `pre_long` / `pre_short`: inngang pris T − 60 min, utgang pris T.
- `post_mom_h12`: signal = fortegn(pris T + 15 min − pris T); inngang T + 15 min, hold 12 barer. (Samme
  vindu som makroregistreringens «bar T+10».)

**E. Lokale klokker — 8 celler.**
- `orb_london_local`: range = de 12 M5-barene fra 08:00 London; inngang ved første close utenfor rangen
  (mid) i bruddretningen; utgang ved siste bar før 17:00 London. `orb_new_york_local`: range fra 08:20
  New York, utgang ved siste bar før 13:30 New York. Én handel per dag.
- Sesjoner long og short: Asia Tokyo 09:00 → London 08:00; London 08:00 → New York 08:20; New York
  08:20 → 13:30. Inngang ved close av første bar ved/etter start, utgang ved close av siste bar før slutt.

## Statistikk og beslutningsregel (låst)

- Per celle ved low: n, snitt netto bps per handel, sd, t_iid = snitt/(sd/√n) og t_dag (standardfeil
  klynget på UTC-inngangsdato); **t = min(t_iid, t_dag)**. Rapporteres også: brutto (spread, uten
  slippage og finansiering) og treffrate, snitt ved low/central/high, snitt per fullt år 2012–2024,
  bjørnefold 2011-09-01 → 2015-12-31, og snitt separat for long- og short-handler.
- 61 celler; Bonferroni ensidig 0,05/61 → **t ≥ 3,149**.
- **GO**: t ≥ 3,149, ≥ 60 % positive fulle år, bjørnefoldens snitt ≥ 0, og — for alle celler der
  signalet velger side (A, B, C, D-post, E-ORB) — positivt snitt for **både** long- og short-handlene.
  Faste-side-celler (D-pre, E-sesjoner) har ikke sidekravet. **LOVENDE**: t ≥ 2,0 med de samme
  tilleggskravene. Ellers NO-GO.
- Beslutning: GO eller LOVENDE → akkurat den cellen bekreftes på urørt VAL før noe bygges; deretter
  maskinlæring bare innenfor den populasjonen (bølge 3). Ingen GO/LOVENDE → pris-alene-mekanismene
  A–E er lukket for intradag-retning i XAUUSD; eneste gjenstående intradagkilde er OANDAs ordre- og
  posisjonsbok (bølge 2, krever operatørvedtak).

## Styrke

Detekterbart snitt ved t = 3,149 ≈ 3,149 × sd/√n. Med sd ≈ 1,25 × E|r| (normaltilnærming; E|r| målt
27.09: 1 t ≈ 12,5–19,7 bps, 4 t ≈ 26–35 bps) og ~3 400 handelsdager: dagscellene på 1 t ≈ 0,8–1,3 bps,
på 4–5 t ≈ 1,8–2,4 bps. A og B har tusener til titusener av handler og ser under 1 bps. Effekter på
størrelse med kostnaden (spread ≈ 2 bps + 2 bps slippage per rundtur ved low) er dermed innenfor det
testen kan skille fra null.

## Kjent begrensning

M5-oppløsning og close-fylling (ingen tick, ingen limit-ordre); rundtall på mid, ikke på bid/ask;
primitivparameterne er 23.09-verdiene, ikke tilpasset; finansieringen er 2026-satsen også for
nullrenteårene (liten for hold ≤ 5 t).

## Kjøring

```
scripts/gx1_capped_run.sh --class producer --mem 10G -- .venv/bin/python -m gx1.scripts.research_intraday_mechanisms_v1 \
  --native-m5-root /home/andre2/GX1_DATA/data/native_xau/XAU_M5_NATIVE_2009_20260701_PAIR_20260927 \
  --cost-policy /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/PROSPECTIVE_COST_POLICY_V1/policy.json \
  --cost-policy-sha256 a48f8e56da21cfa670a80c3b4bfdf735d8ce0b29a25e269184bd5f44fb240a69 \
  --eval-start 2011-06-01T00:00:00Z --read-end-exclusive 2025-06-01T00:00:00Z \
  --out-dir /home/andre2/GX1_RUNS/INTRADAY_MECHANISMS_20260927
```
