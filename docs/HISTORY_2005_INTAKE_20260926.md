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

## Flate vinduer og ny start 2009-06 (målt 27.09, vedtak 27.09)

C0-berikelsen stoppet på `[BASIC_V1_FEATURE_NONFINITE_GAP] _v1_range_z`: 48-barers z-scorer er
udefinerte når alle 48 barer har null spenn. 2006-tapene har én slik M5-episode (langfredag
2009-04-10 17:10 → 04-12 21:40) og 37 M1-episoder, den siste 2009-05-25; ingen etter. Andel
null-spenn-barer M1: 2006 31 %, 2007 20 %, 2008–2009 ~5 %. Berikelsen regner fra tapens første
bar, så en tape som starter før juni 2009 kan ikke bygges uten å finne på verdier (regel 2).
Operatørvedtak: nytt parvedtak `OANDA_PAIR_PRETEST_2009_20260927` fra 2009-06-01. 2008-krakket
faller ut; 2011–15, 2013, 2016, 2018, 2020–21 og 2022 er med.

## Stillestående helgekvoter og stengningskontrakt v2 (målt og vedtatt 27.09)

Den første rebuilden på 2009-tapene stoppet i C0 (M5-lanen) på `[SMC_MTF_OUTPUT_AVAILABILITY_INVALID]`:
alle fire pivoter var like, så SMC-kanalbredden ble null. Årsaken er målt på tapen:

- Tapens egen ukesesjon: siste aktive bar starter fredag 16:55 og første søndag 18:00 New York-tid
  (2013–2026, nesten hver uke; publisert CME/OANDA-metallsesjon).
- Inne i vinduet [fre 17:00, søn 18:00) ligger 1 410 av 1 229 020 M5-barer (0,115 %) og 1 535 M1-barer,
  91 % flate, medianvolum 1. 2011 har 1 020 av dem: én candle hver andre time hele helgen med frossen
  pris (f.eks. 1342,73 hele 22.01.2011). Etter 2013 bare enkeltkvoter rett etter fredagsstengning.

Operatørvedtak 27.09 («filtrer helgevinduet»): nytt parvedtak `OANDA_PAIR_PRETEST_2009_WEEKCLOSED_20260927`
med samme intervall og stengningskontrakt v2
(`oanda_complete_true_scheduled_weekly_closure_excluded_v2`): tapens rader er OANDAs komplette candler
minus dem som starter inne i det planlagte ukevinduet. Kildebitene lagres og beskrives uendret; validatoren
utleder radene fra dem med kontraktens filter. Ingenting syntetiseres — stengningen er fortsatt
kildefravær. v1-tapene er uendret gyldige. Helligdager er ikke dekket av regelen.

## Ikke undersøkt

- Om OANDAs tidlige data har andre handelstider eller sesjonsmønstre enn i dag.
