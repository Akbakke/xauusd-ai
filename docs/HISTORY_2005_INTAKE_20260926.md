# Historikk fra 2006 — innhenting 26.–27.09.2026

Operatørvedtak 26.09: «hent fra 2005». Vedtakene `OANDA_M5_PRETEST_2005_20260926` og
`OANDA_M1_PRETEST_2005_20260926` er bundet i `gx1/contracts/oanda_history_ingest_approval_v1.py`
(bootstrap 2005-01-01 → 2026-07-01, TEST-grensen). Hentet med den kanoniske produsenten
`gx1.scripts.backfill_xauusd_m5_from_oanda` gjennom `scripts/gx1_capped_run.sh --class producer`,
kilde-commit 7fce9960. Logger og revisjoner: `/home/andre2/GX1_RUNS/HISTORY_2005_20260926/`.

## Tapene (målt)

| Tidsramme | Rot under `GX1_DATA/data/native_xau/` | Rader | Første bar | Siste bar | `canonical_rows_sha256` |
|---|---|---|---|---|---|
| M5 | `XAU_M5_NATIVE_2005_20260701_PRETEST_20260926` | 1 446 228 | 2006-03-19 20:25 | 2026-06-30 23:55 | `ae08e57b…99c025` |
| M1 | `XAU_M1_NATIVE_2005_20260701_PRETEST_20260926` | 7 047 777 | 2006-03-19 20:29 | 2026-06-30 23:59 | `67bb206f…7d12f0` |

OANDA har ingen XAU_USD-candles før 2006-03-19 (målt med lesekall 26.09 for både M1 og M5);
vedtakets startdato 2005-01-01 ga derfor tomme bolker før dette, som produsenten tillater.

## Overlapp mot 2019-tapene (målt, rad for rad)

Sammenlignet mot `XAU_{M5,M1}_NATIVE_2019_20260701_PRETEST_20260829` på hele deres spenn
(2019-01-01 23:00 → 2026-06-30):

- Ingen rad i 2019-tapene mangler i de nye. Alle felles rader er verdi-identiske (alle 14 kolonner)
  unntatt to per tidsramme.
- Eneste avvik: **2024-05-20 14:23–17:19 UTC**. 2019-tapene har et hull der (M5: 35 barer,
  M1: 177 barer, median M5-volum 1 220, altså ordinær handel), og barene på hver kant var delvise
  (for eksempel M1 17:20 med volum 5 mot 182 nå). OANDA har fylt vinduet siden 29.08; de nye
  tapene er strengt mer komplette.
- Følge: datasett bygd fra 2019-tapene mangler dette vinduet. Neste rebuild bruker 2006-tapene.

## Kvalitet per år (målt 27.09, `GX1_RUNS/HISTORY_2005_20260926/tape_quality_by_year.json`)

- Ingen high < low eller close utenfor spennet noe år; spread ≤ 0 bare på 1 M5- og 4 M1-barer (2012).
- ~205 hull > 30 min per år er den daglige pausen; enkelte hull på 13–24 t ligger rundt helligdager
  (2006–08, 2011, 2018).
- **Spread (median/p95 bps, M5):** 2006 12,0/26,0 · 2007 6,2/10,9 · 2008 6,9/26,7 · 2009 4,9/17,0 ·
  2010–2021 ≈ 2,0–2,9 · 2022–2026 1,4–1,9. Den bundne kostpolicyen (2 bps per utførelse) undervurderer
  2006–2009; kost må tas fra tapens bid/ask for de årene.
- **Volum (tick-antall, median per M5-bar):** titall i 2006–2011 mot tusenvis nå. Råvolum er en
  epokeproxy og må normaliseres før de tidlige årene går inn i TRAIN.

## Blokkeringer før rebuild (bevist fra kilde 27.09)

1. **Delte vedtak-id-er.** Parprodusenten `gx1.execution.v12_canonical_incremental` krever samme
   `explicit_vedtak_id` på M1- og M5-tapen (`:1161-1183`); godkjenningseieren ga én id per tidsramme.
   **Løst 27.09:** ett parvedtak `OANDA_PAIR_PRETEST_2005_20260927` for begge tidsrammer (samme
   konvensjon som alle tidligere par) og ny henting; de første 2005-tapene over er immutabel historikk
   og kan ikke danne et par.
2. **Ingen TEST-rader** (operatørvedtak 27.09: B — forleng tapene til 2026-09-01 i successor-modus;
   juli–august 2026 blir forseglet TEST, kjeden kjøres uendret). Partapene med felles vedtak:
   `XAU_{M5,M1}_NATIVE_2005_20260701_PAIR_20260927`, rad-identiske med første henting. Rebuild-kjeden kjører alltid full modus og krever at kildens siste rad er
   `--test-end`; tapene slutter ved TEST-grensen. Dataset-wrapperen har en `--pretest-only`-rute som
   kjeden ikke sender videre.
3. **Bootstrap-syklus:** parbygging krever en V4-cache med frosne registerkonstanter og squeeze-sett,
   mens squeeze-fit krever parmanifestet; hvordan syklusen ble brutt sist er ikke registrert.
4. **Squeeze-manifest v4 finnes ikke** (alle sett på disk er v1/v3); refit er obligatorisk og må binde
   samme par og TRAIN-vindu som kjeden.

Tidligste TRAIN-start (estimat, ikke målt): tapestart + ~220 D1-barer oppvarming i parproduktet
(→ historikk tidlig 2007) + 252 lukkede D1-barer før TRAIN (kjedekrav) → TRAIN fra ~feb. 2008.
Nedstrøms lifecycle-v2/random-access-laget har hardkodet TRAIN-start 2021-06-01 og må endres før
native trening, ikke før ukesmålingen.

## Ikke undersøkt

- Om OANDAs tidlige data har andre handelstider eller sesjonsmønstre enn i dag.
