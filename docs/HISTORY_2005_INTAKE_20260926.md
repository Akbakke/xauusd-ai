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

## Ikke undersøkt

- Kvaliteten på de eldste årene (2006–2010): spreader, hull og volum per år er ikke revidert.
- Om OANDAs tidlige data har andre handelstider eller sesjonsmønstre enn i dag.
